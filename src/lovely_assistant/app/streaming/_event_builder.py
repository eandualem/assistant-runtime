"""Pure event factory functions — build streaming event dicts.

No state, no side effects. Each function returns a dict ready for serialization.
"""

from __future__ import annotations

from typing import Any


def make_agent_status_event(status: str) -> dict[str, Any]:
    """Create an agent_status event (started | completed)."""
    return {"type": "agent_status", "status": status}


def make_thinking_delta_event(content: str) -> dict[str, Any]:
    """Create a thinking_delta event."""
    return {"type": "thinking_delta", "content": content}


def make_text_delta_event(content: str) -> dict[str, Any]:
    """Create a text_delta event."""
    return {"type": "text_delta", "content": content}


def make_tool_call_event(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str,
) -> dict[str, Any]:
    """Create a tool_call event (backend or frontend deferred tool)."""
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "call_id": call_id,
    }


def make_tool_result_event(
    tool_name: str,
    output: str,
    call_id: str,
    *,
    duration_ms: float | None = None,
    invalidates: list[str] | None = None,
) -> dict[str, Any]:
    """Create a tool_result event (backend tool execution result)."""
    event: dict[str, Any] = {
        "type": "tool_result",
        "tool_name": tool_name,
        "output": output,
        "call_id": call_id,
    }
    if duration_ms is not None:
        event["duration_ms"] = round(duration_ms, 1)
    if invalidates is not None:
        event["invalidates"] = invalidates
    return event


def make_tool_error_event(
    tool_name: str,
    error: str,
    call_id: str,
) -> dict[str, Any]:
    """Create a tool_error event (tool returned an error result)."""
    return {
        "type": "tool_error",
        "tool_name": tool_name,
        "error": error,
        "call_id": call_id,
    }


def make_tool_status_event(tool_name: str, status: str) -> dict[str, Any]:
    """Create a tool_status event (started | completed | error)."""
    return {"type": "tool_status", "tool_name": tool_name, "status": status}


def make_final_response_event(
    content: str | None,
    model: str,
    *,
    session_id: str | None = None,
    streamed: bool = False,
    thinking_streamed: bool = False,
    error: bool = False,
    error_type: str | None = None,
    usage: dict[str, int] | None = None,
    pending_tool_call: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a final_response event.

    When ``pending_tool_call`` is provided (deferred frontend tool), the
    frontend should execute the tool and send a continuation request with
    the ``call_id`` and the tool result.
    """
    event: dict[str, Any] = {
        "type": "final_response",
        "content": content,
        "model": model,
        "streamed": streamed,
    }
    if session_id is not None:
        event["session_id"] = session_id
    if thinking_streamed:
        event["thinking_streamed"] = True
    if error:
        event["error"] = True
    if error_type is not None:
        event["error_type"] = error_type
    if usage is not None:
        event["usage"] = usage
    if pending_tool_call is not None:
        event["pending_tool_call"] = pending_tool_call
    return event


def make_error_event(
    message: str,
    *,
    error_type: str | None = None,
    terminal: bool = True,
    retry_allowed: bool = False,
) -> dict[str, Any]:
    """Create an error event."""
    event: dict[str, Any] = {
        "type": "error",
        "message": message,
        "terminal": terminal,
        "retry_allowed": retry_allowed,
    }
    if error_type is not None:
        event["error_type"] = error_type
    return event


# --- Debug event factories ---

_MAX_DEBUG_CONTENT = 2000


def _truncate(text: str, max_len: int = _MAX_DEBUG_CONTENT) -> str:
    """Truncate text for debug events."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [{len(text) - max_len} chars truncated]"


def make_debug_request_event(
    session_id: str,
    message: str,
    is_continuation: bool,
    has_machine_state: bool,
    machine_state: dict[str, Any] | None = None,
    image_count: int = 0,
) -> dict[str, Any]:
    """Create a debug_request event — emitted at pipeline entry."""
    return {
        "type": "debug_request",
        "session_id": session_id,
        "message": message,
        "is_continuation": is_continuation,
        "has_machine_state": has_machine_state,
        "machine_state": machine_state,
        "image_count": image_count,
    }


def make_debug_system_prompt_event(
    total_length: int,
    fragment_count: int,
    fragments: list[dict[str, Any]],
    content: str,
) -> dict[str, Any]:
    """Create a debug_system_prompt event — emitted after prompt building."""
    return {
        "type": "debug_system_prompt",
        "total_length": total_length,
        "fragment_count": fragment_count,
        "fragments": fragments,
        "content": content,
    }


def make_debug_history_event(
    message_count: int,
    estimated_tokens: int,
    was_compacted: bool,
    compacted_from: int,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a debug_history event — emitted after history preparation."""
    return {
        "type": "debug_history",
        "message_count": message_count,
        "estimated_tokens": estimated_tokens,
        "was_compacted": was_compacted,
        "compacted_from": compacted_from,
        "messages": messages,
    }


def make_debug_tool_selection_event(
    page: str | None,
    backend_count: int,
    filtered_out: int,
    tool_names: list[str],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a debug_tool_selection event — emitted after tool resolution."""
    return {
        "type": "debug_tool_selection",
        "page": page,
        "backend_count": backend_count,
        "filtered_out": filtered_out,
        "tool_names": tool_names,
        "tools": tools,
    }


def make_debug_agent_config_event(
    model: str | None,
    output_type: str,
    thinking_budget: int | None,
    temperature: float | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Create a debug_agent_config event — emitted after agent creation."""
    event: dict[str, Any] = {
        "type": "debug_agent_config",
        "model": model,
        "output_type": output_type,
        "thinking_budget": thinking_budget,
        "temperature": temperature,
    }
    if session_id is not None:
        event["session_id"] = session_id
    return event


def make_debug_thinking_event(content: str) -> dict[str, Any]:
    """Create a debug_thinking event — accumulated thinking content for trace."""
    return {"type": "debug_thinking", "content": content}


def make_debug_final_response_event(
    content: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Create a debug_final_response event — accumulated response for trace."""
    event: dict[str, Any] = {"type": "debug_final_response", "content": content, "model": model}
    if usage is not None:
        event["usage"] = usage
    return event


def make_debug_usage_event(
    input_tokens: int,
    output_tokens: int,
    cache_read: int,
    cache_write: int,
    requests: int,
    total: int,
) -> dict[str, Any]:
    """Create a debug_usage event — emitted after run completes."""
    return {
        "type": "debug_usage",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "requests": requests,
        "total_tokens": total,
    }


def make_debug_completed_event(
    duration_ms: float,
) -> dict[str, Any]:
    """Create a debug_completed event — emitted in finally block."""
    return {
        "type": "debug_completed",
        "duration_ms": round(duration_ms, 1),
    }
