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
    ToolReturnPart,
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
