"""Pure event factory functions — build SSE event dicts.

No state, no side effects. Each function returns a dict ready for SSE serialization.
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
    """Create a tool_call event (frontend deferred tool)."""
    return {
        "type": "tool_call",
        "tool_name": tool_name,
        "arguments": arguments,
        "call_id": call_id,
    }


def make_tool_status_event(tool_name: str, status: str) -> dict[str, Any]:
    """Create a tool_status event (started | completed | error)."""
    return {"type": "tool_status", "tool_name": tool_name, "status": status}


def make_final_response_event(
    content: str | None,
    model: str,
) -> dict[str, Any]:
    """Create a final_response event."""
    return {
        "type": "final_response",
        "content": content,
        "model": model,
        "streamed": True,
    }


def make_error_event(message: str) -> dict[str, Any]:
    """Create an error event."""
    return {"type": "error", "message": message}
