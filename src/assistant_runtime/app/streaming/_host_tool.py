"""Pending host-tool state on a session.

A host tool (one the host application executes) ends the turn with a
``DeferredToolRequests`` output. The session then remembers the pending call
(``pending_tool_call_id``, ``pending_tool_name``, ``pending_assistant_message_id``)
until the host sends the matching continuation. A new user message that
arrives first abandons the call: its persisted tool entry gets a synthetic
output so the history stays consistent.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.services.tools.interface import ToolService

STALE_HOST_TOOL_OUTPUT = (
    "[Deferred host tool was superseded by a later user turn before its "
    "continuation arrived. The action did not complete.]"
)

_PENDING_KEYS = (
    "pending_tool_call_id",
    "pending_tool_name",
    "pending_assistant_message_id",
    "current_assistant_message_id",
)


async def clear_stale_pending_call(
    sessions: SessionStore, session_id: str, session_context: dict[str, Any]
) -> None:
    """Abandon a pending host-tool call when a new message arrives before its result.

    Host tools can take seconds; a user who sends another message meanwhile
    must not be blocked. The late continuation is rejected by the planner
    because the pending state is gone.
    """
    tool_call_id = session_context.get("pending_tool_call_id")
    if not tool_call_id:
        return

    tool_name = session_context.get("pending_tool_name") or "unknown"
    assistant_message_id = session_context.get("pending_assistant_message_id")
    logger.warning(
        "Clearing pending host tool for new message",
        session_id=session_id,
        pending_tool=tool_name,
        pending_call_id=tool_call_id,
    )
    if assistant_message_id:
        await _mark_superseded(
            sessions,
            session_id,
            session_context,
            assistant_message_id=assistant_message_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
    for key in _PENDING_KEYS:
        session_context.pop(key, None)


async def _mark_superseded(
    sessions: SessionStore,
    session_id: str,
    session_context: dict[str, Any],
    *,
    assistant_message_id: str,
    tool_call_id: str,
    tool_name: str,
) -> None:
    """Persist a synthetic output for the abandoned call on its assistant message."""
    record = session_context["message_index"].get(assistant_message_id)
    if record is None:
        logger.warning(
            "Pending assistant message missing while clearing stale host tool",
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            pending_call_id=tool_call_id,
        )
        return
    segments = superseded_segments(
        record.get("segments"), tool_call_id=tool_call_id, tool_name=tool_name
    )
    if segments is None:
        logger.warning(
            "Stale host tool not found in assistant history; pending state cleared only",
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            pending_call_id=tool_call_id,
            pending_tool=tool_name,
        )
        return
    await sessions.update_message(session_id, assistant_message_id, segments=segments)


def superseded_segments(
    segments: list[dict[str, Any]] | None, *, tool_call_id: str, tool_name: str
) -> list[dict[str, Any]] | None:
    """A copy of ``segments`` with the abandoned call resolved, or None if absent."""
    if not segments:
        return None
    updated = copy.deepcopy(segments)
    for segment in updated:
        if segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict):
                continue
            if tool.get("id") != tool_call_id or tool.get("name") != tool_name:
                continue
            if "output" not in tool:
                tool["output"] = STALE_HOST_TOOL_OUTPUT
            return updated
    return None


def store_pending_call(
    tools: ToolService,
    output: DeferredToolRequests,
    session_context: dict[str, Any],
    *,
    assistant_segments: list[dict[str, Any]] | None = None,
    host_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    """Record the single deferred host-tool call and return its ``final_response`` payload.

    The persisted assistant segments are the source of truth for the call id
    and arguments when they disagree with the run output. ``host_tool_names``
    adds the actions the request declared to the configured host tools.
    """
    calls = list(output.calls)
    if len(calls) != 1:
        raise ValueError(
            f"Host tool protocol violation: expected exactly 1 deferred tool call, got {len(calls)}"
        )
    first = calls[0]
    known = host_tool_names or set()

    def is_host(name: str) -> bool:
        return name in known or tools.is_host_tool(name)

    if not is_host(first.tool_name):
        raise ValueError(
            f"Host tool protocol violation: unknown deferred host tool '{first.tool_name}'"
        )

    pending = _pending_call_from_segments(is_host, assistant_segments)
    if pending is not None:
        if pending["tool_name"] != first.tool_name or pending["call_id"] != first.tool_call_id:
            logger.warning(
                "Deferred host tool metadata drift detected; using assistant history values",
                output_tool_name=first.tool_name,
                output_call_id=first.tool_call_id,
                history_tool_name=pending["tool_name"],
                history_call_id=pending["call_id"],
            )
    else:
        try:
            args = first.args_as_dict()
        except Exception:
            args = {}
        pending = {"tool_name": first.tool_name, "call_id": first.tool_call_id, "arguments": args}

    session_context["pending_tool_call_id"] = pending["call_id"]
    session_context["pending_tool_name"] = pending["tool_name"]
    return pending


def _pending_call_from_segments(
    is_host: Callable[[str], bool], segments: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """The one host-tool entry without an output in persisted assistant segments."""
    if not segments:
        return None
    pending = [
        tool
        for segment in segments
        if segment.get("kind") == "tool_group"
        for tool in segment.get("tools", [])
        if isinstance(tool, dict) and is_host(str(tool.get("name", ""))) and "output" not in tool
    ]
    if not pending:
        return None
    if len(pending) != 1:
        raise ValueError(
            "Host tool protocol violation: "
            f"expected exactly 1 pending host tool in assistant history, got {len(pending)}"
        )
    tool = pending[0]
    return {
        "tool_name": str(tool.get("name", "")),
        "call_id": str(tool.get("id", "")),
        "arguments": tool.get("input") if isinstance(tool.get("input"), dict) else {},
    }
