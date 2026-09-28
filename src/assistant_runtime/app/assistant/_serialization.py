"""Conversation tree serialization helpers.

A message row keeps its part of the conversation as Pydantic AI messages
(``model_messages``), which later turns replay unchanged, and, for assistant
rows, ``segments``: the display form and the index of tool calls. Rows stored
before ``model_messages`` existed are replayed from their content and segments.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from pydantic import ConfigDict, TypeAdapter
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from assistant_runtime.host_context import Attachment

MessageRecord = dict[str, Any]
SteeringRecord = dict[str, Any]

_TOOL_OUTPUT_ADAPTER = TypeAdapter(Any, config=ConfigDict(ser_json_bytes="base64"))


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


def dump_model_messages(messages: Sequence[ModelMessage]) -> list[dict[str, Any]]:
    """The stored form of ``messages``: Pydantic AI's serialisation, as JSON values.

    Instructions are left out, since every run sends its own. Media in tool
    output is redacted as in the segments (``normalize_tool_output_for_storage``).
    """
    stored: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            message = replace(
                message,
                instructions=None,
                parts=[
                    replace(
                        part,
                        content=normalize_tool_output_for_storage(part.tool_name, part.content),
                    )
                    if isinstance(part, ToolReturnPart)
                    else part
                    for part in message.parts
                ],
            )
        stored.append(message)
    return ModelMessagesTypeAdapter.dump_python(stored, mode="json")


def load_model_messages(stored: list[dict[str, Any]]) -> list[ModelMessage]:
    """Messages from their stored form (see ``dump_model_messages``)."""
    return ModelMessagesTypeAdapter.validate_python(stored)


def user_prompt_messages(
    prompt: str | Sequence[UserContent], timestamp: datetime
) -> list[dict[str, Any]]:
    """The stored form of a user row: the request its prompt was sent in."""
    return dump_model_messages(
        [ModelRequest(parts=[UserPromptPart(content=prompt, timestamp=timestamp)])]
    )


def attachment_content(attachment: Attachment) -> UserContent:
    """A reference attachment as native Pydantic AI user content."""
    if attachment.text is not None:
        return f"[{attachment.name or 'attachment'}]\n{attachment.text}"
    if attachment.data_uri is not None:
        return BinaryContent.from_data_uri(attachment.data_uri)
    assert attachment.url is not None
    if attachment.kind == "image":
        return ImageUrl(url=attachment.url)
    return DocumentUrl(url=attachment.url)


def path_records_to_model_history(messages: list[MessageRecord]) -> list[ModelMessage]:
    """Expand tree message rows into the linear ModelMessage history seen by the LLM.

    A row with ``model_messages`` contributes exactly those messages.
    """
    history: list[ModelMessage] = []
    for message in messages:
        role = message.get("role")
        stored = message.get("model_messages")
        if stored is not None:
            history.extend(load_model_messages(stored))
        elif role == "user":
            history.append(_user_record_to_request(message, with_host_context=True))
        elif role == "assistant":
            history.extend(_assistant_record_to_messages(message))
        elif role == "host" and message.get("content"):
            # A host message reaches the model as its text only; its components are for people.
            history.append(_user_record_to_request(message))
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

        if role == "host":
            display.append(
                {
                    "id": message["id"],
                    "parent_id": message.get("parent_id"),
                    "role": "host",
                    "text": message.get("content", ""),
                    "segments": list(message.get("segments") or []),
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
        created_at = (
            _coerce_datetime(steering.get("created_at")) or delivered_at or datetime.now(UTC)
        )
        display_items = steering_records_to_display([steering])
        if not display_items:
            continue
        timeline.append(
            (delivered_at or created_at, created_at, len(path_messages) + index, display_items[0])
        )

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
    *,
    interrupted_status: str = "superseded",
) -> tuple[str, list[dict[str, Any]], datetime | None]:
    """Extract assistant content + canonicalized segments from a single turn's messages.

    ``interrupted_status`` is the ``status`` recorded on calls whose result is
    an ``interrupted`` outcome: ``cancelled`` when the turn was cancelled or
    stopped, ``superseded`` when core closed an older call.
    """
    tool_results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                result = {"output": normalize_tool_output_for_storage(part.tool_name, part.content)}
                # Existing records imply success. Preserve native failures and
                # denials explicitly so reloaded history keeps their semantics;
                # ``status`` says how a non-successful action was resolved.
                if part.outcome != "success":
                    result["outcome"] = part.outcome
                if part.outcome == "failed":
                    result["status"] = "failed"
                elif part.outcome == "interrupted":
                    result["status"] = interrupted_status
                tool_results[part.tool_call_id] = result

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
                    "input": part.args_as_dict(),
                }
                if part.tool_call_id in tool_results:
                    tool_entry.update(tool_results[part.tool_call_id])
                if segments and segments[-1]["kind"] == "tool_group":
                    segments[-1]["tools"].append(tool_entry)
                else:
                    segments.append({"kind": "tool_group", "tools": [tool_entry]})

    canonical = canonicalize_assistant_segments(segments)
    return assistant_segments_to_text(canonical), canonical, timestamp


def build_steering_request(steering_records: list[SteeringRecord]) -> ModelRequest:
    """Build a model request with each steering message's text as it was sent.

    A steering message's reference attachments follow its text as native
    content. The request's metadata (never sent to the model) names the records.
    """
    parts = []
    for steering in steering_records:
        text = steering.get("content", "")
        attachments = [
            attachment_content(Attachment.model_validate(item))
            for item in steering.get("attachments") or []
        ]
        parts.append(
            UserPromptPart(
                content=[text, *attachments] if attachments else text,
                timestamp=_coerce_datetime(steering.get("delivered_at"))
                or _coerce_datetime(steering.get("created_at"))
                or datetime.now(UTC),
            )
        )
    return ModelRequest(
        parts=parts, metadata={"steering_ids": [steering["id"] for steering in steering_records]}
    )


def normalize_tool_output_for_storage(tool_name: str, content: Any) -> Any:
    """Redact media and encode native tool values for JSON-backed session storage."""
    redacted = _omit_tool_output_media(tool_name, content)
    # Native tools can return models, dataclasses, dates and other values that
    # Pydantic AI accepts but the database's plain JSON serializer does not.
    # Match upstream's JSON-mode and alias handling without changing model input.
    return _TOOL_OUTPUT_ADAPTER.dump_python(redacted, mode="json", by_alias=True)


def _omit_tool_output_media(tool_name: str, content: Any) -> Any:
    """Preserve the runtime's media omission policy before JSON conversion."""
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
        return {key: _omit_tool_output_media(tool_name, value) for key, value in content.items()}
    if isinstance(content, (list, tuple)):
        return [_omit_tool_output_media(tool_name, item) for item in content]
    return content


def sanitize_image_tool_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Replace ``look_at_screen`` results with a placeholder once the model has seen them."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == "look_at_screen":
                part.content = "[Inspected current screen]"
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
                        outcome=tool.get("outcome", "success"),
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
    """Build continuation history that separates completed tools from pending ones.

    Pydantic AI's ``_handle_deferred_tool_results`` finds the **last**
    ``ModelResponse`` in the history and validates that every tool call in it
    has a matching entry in ``tool_call_results``.  When completed backend
    tools and a pending host tool share the same ``ModelResponse``, the
    backend tool's 'skip' entry (added from ``last_model_request``) creates a
    set-equality mismatch (more IDs in ``tool_call_results`` than in the
    response's tool calls).

    The fix: emit completed tools in one ``ModelResponse`` + ``ModelRequest``
    pair, then emit pending (no-output) tools in a **separate trailing**
    ``ModelResponse``.  This way ``_handle_deferred_tool_results`` sees only the
    pending tool in ``last_model_response`` and ``last_model_request`` is ``None``
    — no 'skip' entries are added and the sets match.
    """
    segments = canonicalize_assistant_segments(message.get("segments"))
    if not segments and message.get("content"):
        segments = [{"kind": "text", "text": message["content"]}]

    # Collect all parts, separating completed tools from pending ones.
    content_parts: list[Any] = []  # thinking, text
    completed_tool_calls: list[ToolCallPart] = []
    completed_tool_returns: list[ToolReturnPart] = []
    pending_tool_calls: list[ToolCallPart] = []
    timestamp = _coerce_datetime(message.get("created_at")) or datetime.now(UTC)

    for segment in segments:
        kind = segment["kind"]
        if kind == "thinking":
            content_parts.append(ThinkingPart(content=segment["text"]))
        elif kind == "text":
            content_parts.append(TextPart(content=segment["text"]))
        elif kind == "tool_group":
            for tool in segment.get("tools", []):
                call_part = ToolCallPart(
                    tool_name=str(tool.get("name", "")),
                    args=tool.get("input") if isinstance(tool.get("input"), dict) else {},
                    tool_call_id=str(tool.get("id", "")),
                )
                if "output" in tool:
                    completed_tool_calls.append(call_part)
                    completed_tool_returns.append(
                        ToolReturnPart(
                            tool_name=str(tool.get("name", "")),
                            content=tool.get("output"),
                            tool_call_id=str(tool.get("id", "")),
                            timestamp=timestamp,
                            outcome=tool.get("outcome", "success"),
                        )
                    )
                else:
                    pending_tool_calls.append(call_part)

    if not content_parts and not completed_tool_calls and not pending_tool_calls:
        return []

    usage = _request_usage(message.get("usage"))
    result: list[ModelMessage] = []

    # Completed tools (+ content) go in the first ModelResponse/ModelRequest.
    completed_response_parts = [*content_parts, *completed_tool_calls]
    if completed_response_parts:
        result.append(
            ModelResponse(parts=completed_response_parts, usage=usage, timestamp=timestamp)
        )
    if completed_tool_returns:
        result.append(ModelRequest(parts=completed_tool_returns, timestamp=timestamp))

    # Pending tools go in a separate trailing ModelResponse so that
    # _handle_deferred_tool_results sees only these in last_model_response.
    if pending_tool_calls:
        result.append(ModelResponse(parts=pending_tool_calls, timestamp=timestamp))

    return result


def _user_record_to_request(
    message: MessageRecord, *, with_host_context: bool = False
) -> ModelRequest:
    content = message.get("content", "")
    timestamp = _coerce_datetime(message.get("created_at"))
    prompt: str | list[UserContent] = content
    if with_host_context:
        # The block the message was sent with, ahead of its text (see build_user_prompt).
        blocks: list[UserContent] = [
            str(segment.get("text", ""))
            for segment in message.get("segments") or []
            if isinstance(segment, dict) and segment.get("kind") == "host_context"
        ]
        if blocks:
            prompt = [*blocks, content] if content else blocks
    return ModelRequest(
        parts=[UserPromptPart(content=prompt, timestamp=timestamp or datetime.now(UTC))],
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
