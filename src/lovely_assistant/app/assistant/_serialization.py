"""Conversation tree serialization helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

MessageRecord = dict[str, Any]
SteeringRecord = dict[str, Any]

_STEERING_PREFIX = "Additional user steering while you were working:\n"


def canonicalize_assistant_segments(segments: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Merge adjacent content segments while preserving tool boundaries."""
    if not segments:
        return []

    normalized: list[dict[str, Any]] = []
    for segment in segments:
        kind = segment.get("kind")
        if kind not in {"thinking", "text", "tool_group"}:
            continue
        if kind == "tool_group":
            tools = [dict(tool) for tool in segment.get("tools", []) if isinstance(tool, dict)]
            if not tools:
                continue
            normalized.append({"kind": "tool_group", "tools": tools})
            continue

        text = str(segment.get("text", ""))
        if not text:
            continue
        if normalized and normalized[-1]["kind"] == kind:
            normalized[-1]["text"] += text
            continue
        normalized.append({"kind": kind, "text": text})

    return normalized


def path_records_to_model_history(messages: list[MessageRecord]) -> list[ModelMessage]:
    """Expand tree message rows into the linear ModelMessage history seen by the LLM."""
    history: list[ModelMessage] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            history.append(_user_record_to_request(message))
        elif role == "assistant":
            history.extend(_assistant_record_to_messages(message))
    return history


def tree_messages_to_display(messages: list[MessageRecord]) -> list[dict[str, Any]]:
    """Serialize a linear root-to-leaf path for `/messages`."""
    display: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            display.append(
                {
                    "id": message["id"],
                    "parent_id": message.get("parent_id"),
                    "role": "user",
                    "text": message.get("content", ""),
                    "message_type": message.get("message_type", "standard"),
                    "timestamp": _iso(message.get("created_at")),
                }
            )
            continue

        if role == "assistant":
            segments = _with_segment_metadata(
                canonicalize_assistant_segments(message.get("segments"))
            )
            display.append(
                {
                    "id": message["id"],
                    "parent_id": message.get("parent_id"),
                    "role": "assistant",
                    "text": assistant_segments_to_text(segments),
                    "segments": segments,
                    "usage": message.get("usage"),
                    "timestamp": _iso(message.get("created_at")),
                }
            )
    return display


def steering_records_to_display(steering_records: list[SteeringRecord]) -> list[dict[str, Any]]:
    """Serialize delivered/promoted steering records for `/messages`."""
    display: list[dict[str, Any]] = []
    for steering in steering_records:
        status = steering.get("status")
        if status not in {"delivered", "promoted"}:
            continue
        timestamp = _iso(steering.get("delivered_at")) or _iso(steering.get("created_at"))
        display.append(
            {
                "id": steering["id"],
                "role": "steering",
                "text": steering.get("content", ""),
                "message_type": "steering",
                "status": status,
                "timestamp": timestamp,
            }
        )
    return display


def merge_display_messages(
    path_messages: list[MessageRecord],
    steering_records: list[SteeringRecord],
) -> list[dict[str, Any]]:
    """Merge a root-to-leaf message path with delivered steering display entries."""
    timeline: list[tuple[datetime, datetime, int, dict[str, Any]]] = []

    for index, message in enumerate(path_messages):
        created_at = _coerce_datetime(message.get("created_at")) or datetime.now(UTC)
        display = tree_messages_to_display([message])[0]
        timeline.append((created_at, created_at, index, display))

    for index, steering in enumerate(steering_records):
        status = steering.get("status")
        if status not in {"delivered", "promoted"}:
            continue
        delivered_at = _coerce_datetime(steering.get("delivered_at")) or _coerce_datetime(
            steering.get("created_at")
        )
        created_at = _coerce_datetime(steering.get("created_at")) or delivered_at or datetime.now(UTC)
        display_items = steering_records_to_display([steering])
        if not display_items:
            continue
        timeline.append((delivered_at or created_at, created_at, len(path_messages) + index, display_items[0]))

    timeline.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[-1] for item in timeline]


def tree_messages_to_tree(messages: list[MessageRecord]) -> list[dict[str, Any]]:
    """Serialize all session messages for `/tree`."""
    return [
        {
            "id": message["id"],
            "parent_id": message.get("parent_id"),
            "role": message.get("role"),
            "message_type": message.get("message_type", "standard"),
            "content": message.get("content", ""),
            "created_at": _iso(message.get("created_at")),
        }
        for message in messages
    ]


def assistant_segments_to_text(segments: list[dict[str, Any]] | None) -> str:
    """Concatenate text segments for display / summary fields."""
    if not segments:
        return ""
    return "".join(segment.get("text", "") for segment in segments if segment.get("kind") == "text")


def build_assistant_message_content(
    messages: list[ModelMessage],
) -> tuple[str, list[dict[str, Any]], datetime | None]:
    """Extract assistant content + canonicalized segments from a single turn's messages."""
    tool_results: dict[str, Any] = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                tool_results[part.tool_call_id] = normalize_tool_output_for_storage(
                    part.tool_name,
                    part.content,
                )

    segments: list[dict[str, Any]] = []
    timestamp: datetime | None = None
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        if timestamp is None:
            timestamp = message.timestamp
        for part in message.parts:
            if isinstance(part, ThinkingPart):
                _append_content_segment(segments, "thinking", part.content)
            elif isinstance(part, TextPart):
                _append_content_segment(segments, "text", part.content)
            elif isinstance(part, ToolCallPart):
                tool_entry = {
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args if isinstance(part.args, dict) else {},
                }
                if part.tool_call_id in tool_results:
                    tool_entry["output"] = tool_results[part.tool_call_id]
                if segments and segments[-1]["kind"] == "tool_group":
                    segments[-1]["tools"].append(tool_entry)
                else:
                    segments.append({"kind": "tool_group", "tools": [tool_entry]})

    canonical = canonicalize_assistant_segments(segments)
    return assistant_segments_to_text(canonical), canonical, timestamp


def build_steering_request(steering_records: list[SteeringRecord]) -> ModelRequest:
    """Build a model request that frames steering distinctly from user messages."""
    parts = [
        UserPromptPart(
            content=f"{_STEERING_PREFIX}{steering.get('content', '')}",
            timestamp=_coerce_datetime(steering.get("delivered_at"))
            or _coerce_datetime(steering.get("created_at"))
            or datetime.now(UTC),
        )
        for steering in steering_records
    ]
    return ModelRequest(parts=parts)


def normalize_tool_output_for_storage(tool_name: str, content: Any) -> Any:
    """Strip binary / multimodal payloads so persisted tool outputs stay serializable."""
    if tool_name == "look_at_screen":
        return "[Inspected current screen]"
    if isinstance(content, BinaryContent):
        return "[Binary content omitted]"
    try:
        import pydantic_ai.messages as pydantic_messages
    except ImportError:  # pragma: no cover
        pydantic_messages = None  # type: ignore[assignment]
    file_url_cls = getattr(pydantic_messages, "FileUrl", None)
    if file_url_cls is not None and isinstance(content, file_url_cls):
        result: dict[str, Any] = {"url": content.url}
        if hasattr(content, "kind"):
            result["kind"] = content.kind
        return result
    if isinstance(content, dict):
        return {key: normalize_tool_output_for_storage(tool_name, value) for key, value in content.items()}
    if isinstance(content, list):
        return [normalize_tool_output_for_storage(tool_name, item) for item in content]
    return content


def sanitize_image_tool_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip binary screen-inspection payloads from a ModelMessage history."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "look_at_screen":
                part.content = "[Inspected current screen]"
        message.parts = [part for part in message.parts if not _is_synthetic_binary_user_prompt(part)]
    return messages


def _assistant_record_to_messages(message: MessageRecord) -> list[ModelMessage]:
    segments = canonicalize_assistant_segments(message.get("segments"))
    if not segments and message.get("content"):
        segments = [{"kind": "text", "text": message["content"]}]

    response_groups: list[list[Any]] = []
    current_parts: list[Any] = []
    request_messages: list[ModelRequest | None] = []

    for segment in segments:
        kind = segment["kind"]
        if kind == "thinking":
            current_parts.append(ThinkingPart(content=segment["text"]))
            continue
        if kind == "text":
            current_parts.append(TextPart(content=segment["text"]))
            continue

        if kind == "tool_group":
            for tool in segment.get("tools", []):
                current_parts.append(
                    ToolCallPart(
                        tool_name=str(tool.get("name", "")),
                        args=tool.get("input") if isinstance(tool.get("input"), dict) else {},
                        tool_call_id=str(tool.get("id", "")),
                    )
                )
            response_groups.append(current_parts)
            current_parts = []

            tool_returns = []
            for tool in segment.get("tools", []):
                if "output" not in tool:
                    continue
                tool_returns.append(
                    ToolReturnPart(
                        tool_name=str(tool.get("name", "")),
                        content=tool.get("output"),
                        tool_call_id=str(tool.get("id", "")),
                        timestamp=_coerce_datetime(message.get("created_at")) or datetime.now(UTC),
                    )
                )
            request_messages.append(
                ModelRequest(
                    parts=tool_returns,
                    timestamp=_coerce_datetime(message.get("created_at")),
                )
                if tool_returns
                else None
            )

    if current_parts:
        response_groups.append(current_parts)

    usage = _request_usage(message.get("usage"))
    timestamp = _coerce_datetime(message.get("created_at")) or datetime.now(UTC)

    model_messages: list[ModelMessage] = []
    for index, parts in enumerate(response_groups):
        response_usage = usage if index == len(response_groups) - 1 else RequestUsage()
        model_messages.append(
            ModelResponse(
                parts=parts,
                usage=response_usage,
                timestamp=timestamp,
            )
        )
        if index < len(request_messages) and request_messages[index] is not None:
            model_messages.append(request_messages[index])  # type: ignore[arg-type]
    return model_messages


def assistant_record_to_flat_messages(message: MessageRecord) -> list[ModelMessage]:
    """Build a flat ModelResponse + ModelRequest for continuation history.

    Unlike _assistant_record_to_messages (which splits tool groups into separate
    ModelResponse/ModelRequest pairs), this creates a SINGLE ModelResponse with
    all parts and a SINGLE ModelRequest with all resolved tool results. This
    matches what Pydantic AI's _handle_deferred_tool_results expects when
    resuming with DeferredToolResults.
    """
    segments = canonicalize_assistant_segments(message.get("segments"))
    if not segments and message.get("content"):
        segments = [{"kind": "text", "text": message["content"]}]

    response_parts: list[Any] = []
    tool_return_parts: list[ToolReturnPart] = []
    timestamp = _coerce_datetime(message.get("created_at")) or datetime.now(UTC)

    for segment in segments:
        kind = segment["kind"]
        if kind == "thinking":
            response_parts.append(ThinkingPart(content=segment["text"]))
        elif kind == "text":
            response_parts.append(TextPart(content=segment["text"]))
        elif kind == "tool_group":
            for tool in segment.get("tools", []):
                response_parts.append(
                    ToolCallPart(
                        tool_name=str(tool.get("name", "")),
                        args=tool.get("input") if isinstance(tool.get("input"), dict) else {},
                        tool_call_id=str(tool.get("id", "")),
                    )
                )
                if "output" in tool:
                    tool_return_parts.append(
                        ToolReturnPart(
                            tool_name=str(tool.get("name", "")),
                            content=tool.get("output"),
                            tool_call_id=str(tool.get("id", "")),
                            timestamp=timestamp,
                        )
                    )

    if not response_parts:
        return []

    usage = _request_usage(message.get("usage"))
    result: list[ModelMessage] = [ModelResponse(parts=response_parts, usage=usage, timestamp=timestamp)]
    if tool_return_parts:
        result.append(ModelRequest(parts=tool_return_parts, timestamp=timestamp))
    return result


def _user_record_to_request(message: MessageRecord) -> ModelRequest:
    content = message.get("content", "")
    timestamp = _coerce_datetime(message.get("created_at"))
    return ModelRequest(
        parts=[UserPromptPart(content=content, timestamp=timestamp or datetime.now(UTC))],
        timestamp=timestamp,
    )


def _request_usage(data: dict[str, Any] | None) -> RequestUsage:
    if not data:
        return RequestUsage()
    return RequestUsage(
        input_tokens=int(data.get("input_tokens", 0) or 0),
        output_tokens=int(data.get("output_tokens", 0) or 0),
    )


def _append_content_segment(segments: list[dict[str, Any]], kind: str, text: str) -> None:
    if not text:
        return
    if segments and segments[-1]["kind"] == kind:
        segments[-1]["text"] += text
        return
    segments.append({"kind": kind, "text": text})


def _is_synthetic_binary_user_prompt(part: Any) -> bool:
    return (
        isinstance(part, UserPromptPart)
        and isinstance(part.content, list)
        and any(isinstance(item, BinaryContent) for item in part.content)
    )


def _with_segment_metadata(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        item = dict(segment)
        item["segment_id"] = f"segment_{index}"
        item["segment_index"] = index
        enriched.append(item)
    return enriched


def _iso(value: Any) -> str | None:
    dt = _coerce_datetime(value)
    return dt.isoformat() if dt is not None else None


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
