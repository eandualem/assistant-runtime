"""Message history serialization helpers.

Handles round-tripping Pydantic AI's typed ModelMessage list through JSONB storage.
Ported from arclio-assistant's session model — fixes the FileUrl reconstruction gotcha
where Pydantic AI auto-converts dicts with 'url' keys into FileUrl subclasses on
deserialization.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Serialize typed ModelMessage list to JSON-compatible dicts for JSONB storage."""
    json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
    import json

    return json.loads(json_bytes)


def deserialize_messages(data: list[dict[str, Any]]) -> list[ModelMessage]:
    """Deserialize JSONB data back to typed ModelMessage list.

    Applies sanitization to fix rogue FileUrl reconstructions in tool return content.
    """
    if not data:
        return []
    messages = ModelMessagesTypeAdapter.validate_python(data)
    return _sanitize_history(messages)


def _sanitize_tool_return_content(content: Any) -> Any:
    """Recursively convert rogue FileUrl objects back to plain dicts.

    Pydantic AI's ToolReturnContent type auto-reconstructs dicts containing
    a 'url' key as ImageUrl/AudioUrl/DocumentUrl/VideoUrl during deserialization.
    Tool results are arbitrary data and should not contain multi-modal content
    objects — when re-serialized, these objects fail if the URL lacks a
    recognizable file extension.
    """
    try:
        from pydantic_ai.messages import FileUrl
    except ImportError:
        return content

    if isinstance(content, FileUrl):
        result: dict[str, Any] = {"url": content.url}
        if hasattr(content, "kind"):
            result["kind"] = content.kind
        return result
    if isinstance(content, dict):
        return {k: _sanitize_tool_return_content(v) for k, v in content.items()}
    if isinstance(content, list):
        return [_sanitize_tool_return_content(item) for item in content]
    return content


def _sanitize_history(history: list[Any]) -> list[Any]:
    """Sanitize deserialized history to fix rogue type reconstructions.

    Walks all ToolReturnPart.content values and converts any FileUrl
    subclass objects back to plain dicts.
    """
    sanitized_count = 0
    for msg in history:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    original = part.content
                    sanitized = _sanitize_tool_return_content(original)
                    if sanitized is not original:
                        part.content = sanitized
                        sanitized_count += 1
    if sanitized_count > 0:
        logger.info(
            f"Sanitized {sanitized_count} tool return part(s) with rogue FileUrl reconstructions"
        )
    return history


def messages_to_display_format(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """Convert ModelMessage list to display-ready dicts for the frontend.

    Two-pass conversion:
    1. Collect ToolReturnPart results keyed by tool_call_id
    2. Build display messages with ordered segments, collapsing consecutive
       assistant turns (ModelResponse messages separated by tool-return-only
       ModelRequests) into single entries
    """
    # Pass 1: collect tool return results
    tool_results: dict[str, Any] = {}
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    tool_results[part.tool_call_id] = part.content

    # Pass 2: build display messages with ordered segments
    display: list[dict[str, Any]] = []
    pending_segments: list[dict[str, Any]] = []
    pending_timestamp: str | None = None

    def _flush_assistant() -> None:
        nonlocal pending_segments, pending_timestamp
        if not pending_segments:
            return
        display.append(_build_assistant_entry(pending_segments, pending_timestamp))
        pending_segments = []
        pending_timestamp = None

    for msg in messages:
        if isinstance(msg, ModelRequest):
            user_texts: list[str] = []
            has_user_prompt = False
            for part in msg.parts:
                if not isinstance(part, UserPromptPart):
                    continue
                has_user_prompt = True
                if isinstance(part.content, str):
                    user_texts.append(part.content)
                elif isinstance(part.content, list):
                    for item in part.content:
                        if isinstance(item, str):
                            user_texts.append(item)
            if not has_user_prompt:
                continue
            if not user_texts:
                user_texts.append("[Image attachment]")
            _flush_assistant()
            entry: dict[str, Any] = {
                "role": "user",
                "text": "\n".join(user_texts),
            }
            if msg.timestamp is not None:
                entry["timestamp"] = msg.timestamp.isoformat()
            display.append(entry)

        elif isinstance(msg, ModelResponse):
            if pending_timestamp is None and msg.timestamp is not None:
                pending_timestamp = msg.timestamp.isoformat()

            for part in msg.parts:
                if isinstance(part, ThinkingPart):
                    pending_segments.append({"kind": "thinking", "text": part.content})
                elif isinstance(part, TextPart):
                    pending_segments.append({"kind": "text", "text": part.content})
                elif isinstance(part, ToolCallPart):
                    tc: dict[str, Any] = {
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "input": part.args if isinstance(part.args, dict) else {},
                    }
                    if part.tool_call_id in tool_results:
                        tc["output"] = tool_results[part.tool_call_id]
                    if pending_segments and pending_segments[-1]["kind"] == "tool_group":
                        pending_segments[-1]["tools"].append(tc)
                    else:
                        pending_segments.append({"kind": "tool_group", "tools": [tc]})

    _flush_assistant()
    return display


def _build_assistant_entry(segments: list[dict[str, Any]], timestamp: str | None) -> dict[str, Any]:
    """Build an assistant display entry from ordered segments.

    Produces both the ordered ``segments`` array (for position-aware rendering)
    and backward-compatible flat ``text``/``thinking``/``tool_calls`` fields.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for seg in segments:
        if seg["kind"] == "text":
            text_parts.append(seg["text"])
        elif seg["kind"] == "thinking":
            thinking_parts.append(seg["text"])
        elif seg["kind"] == "tool_group":
            tool_calls.extend(seg["tools"])

    entry: dict[str, Any] = {
        "role": "assistant",
        "text": "".join(text_parts),
        "segments": segments,
    }
    if thinking_parts:
        entry["thinking"] = "".join(thinking_parts)
    if tool_calls:
        entry["tool_calls"] = tool_calls
    if timestamp is not None:
        entry["timestamp"] = timestamp
    return entry
