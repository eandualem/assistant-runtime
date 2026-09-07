"""Pending host-tool state on a session.

A host tool (one the host application executes) ends the turn with a
``DeferredToolRequests`` output. The session then remembers the pending call
(``pending_tool_call_id``, ``pending_tool_name``, ``pending_assistant_message_id``,
and ``pending_tool_batch``, the ids of every host call in that response,
stored on the session row through ``SessionStore``) until the host sends the
matching continuation. Several host calls in one response are handed to the
host one at a time, in order; the model resumes once all have results. A new
user message that arrives first abandons them: their persisted tool entries
get a synthetic ``superseded`` result so the history stays consistent and a
late continuation is rejected.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests

from assistant_runtime.app.assistant._stale_tools import resolve_tool_entry

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.services.tools.interface import ToolService

STALE_HOST_TOOL_OUTPUT = (
    "[Deferred host tool was superseded by a later user turn before its "
    "continuation arrived. Its result was never recorded.]"
)


async def clear_stale_pending_call(
    sessions: SessionStore, session_id: str, session_context: dict[str, Any]
) -> None:
    """Abandon a pending host-tool call when a new message arrives before its result.

    Host tools can take seconds; a user who sends another message meanwhile
    must not be blocked. The late continuation is rejected by the planner
    because the pending state is gone, in memory and on the row.
    """
    tool_call_id = session_context.get("pending_tool_call_id")
    if not tool_call_id:
        return

    tool_name = session_context.get("pending_tool_name") or "unknown"
    assistant_message_id = session_context.get("pending_assistant_message_id")
    batch = [str(c) for c in (session_context.get("pending_tool_batch") or [tool_call_id])]
    logger.warning(
        "Clearing pending host tool for new message",
        session_id=session_id,
        pending_tool=tool_name,
        pending_call_id=tool_call_id,
        batch=batch,
    )
    if assistant_message_id:
        await _mark_superseded(
            sessions,
            session_id,
            session_context,
            assistant_message_id=assistant_message_id,
            tool_call_ids=set(batch) | {str(tool_call_id)},
        )
    await sessions.clear_pending_action(session_id)
    session_context.pop("current_assistant_message_id", None)


async def _mark_superseded(
    sessions: SessionStore,
    session_id: str,
    session_context: dict[str, Any],
    *,
    assistant_message_id: str,
    tool_call_ids: set[str],
) -> None:
    """Persist synthetic outputs for the abandoned calls on their assistant message."""
    record = session_context["message_index"].get(assistant_message_id)
    if record is None:
        logger.warning(
            "Pending assistant message missing while clearing stale host tool",
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            pending_call_ids=sorted(tool_call_ids),
        )
        return
    segments = superseded_segments(record.get("segments"), tool_call_ids=tool_call_ids)
    if segments is None:
        logger.warning(
            "Stale host tool not found in assistant history; pending state cleared only",
            session_id=session_id,
            assistant_message_id=assistant_message_id,
            pending_call_ids=sorted(tool_call_ids),
        )
        return
    await sessions.update_message(session_id, assistant_message_id, segments=segments)


def superseded_segments(
    segments: list[dict[str, Any]] | None, *, tool_call_ids: set[str]
) -> list[dict[str, Any]] | None:
    """A copy of ``segments`` with the abandoned, unanswered calls resolved; None if none found."""
    if not segments:
        return None
    updated = copy.deepcopy(segments)
    found = False
    for segment in updated:
        if segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict) or str(tool.get("id")) not in tool_call_ids:
                continue
            found = True
            if "output" not in tool:
                resolve_tool_entry(tool, output=STALE_HOST_TOOL_OUTPUT, status="superseded")
    return updated if found else None


def pending_call_payloads(
    tools: ToolService,
    output: DeferredToolRequests,
    *,
    assistant_segments: list[dict[str, Any]] | None = None,
    host_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate the deferred host-tool calls and return their payloads, in call order.

    A model may ask for several host actions in one response; they are handed
    to the host one at a time, so the first payload is the ``final_response``
    ``pending_tool_call`` and the rest follow as continuations arrive. The
    persisted assistant segments are the source of truth for arguments when
    they disagree with the run output. ``host_tool_names`` adds the actions
    the request declared to the configured host tools. The caller records the
    calls on the session through ``SessionStore``.
    """
    calls = list(output.calls)
    if not calls:
        raise ValueError("Host tool protocol violation: deferred output without tool calls")
    known = host_tool_names or set()

    def is_host(name: str) -> bool:
        return name in known or tools.is_host_tool(name)

    for call in calls:
        if not is_host(call.tool_name):
            raise ValueError(
                f"Host tool protocol violation: unknown deferred host tool '{call.tool_name}'"
            )

    stored = _pending_calls_from_segments(is_host, assistant_segments)
    if len(calls) == 1 and len(stored) == 1 and calls[0].tool_call_id not in stored:
        # The persisted history is the source of truth when the run output drifted.
        (entry,) = stored.values()
        logger.warning(
            "Deferred host tool metadata drift detected; using assistant history values",
            output_tool_name=calls[0].tool_name,
            output_call_id=calls[0].tool_call_id,
            history_tool_name=entry["tool_name"],
            history_call_id=entry["call_id"],
        )
        return [entry]
    payloads: list[dict[str, Any]] = []
    for call in calls:
        entry = stored.get(call.tool_call_id)
        if entry is not None:
            if entry["tool_name"] != call.tool_name:
                logger.warning(
                    "Deferred host tool metadata drift detected; using assistant history values",
                    output_tool_name=call.tool_name,
                    call_id=call.tool_call_id,
                    history_tool_name=entry["tool_name"],
                )
            payloads.append(entry)
            continue
        try:
            args = call.args_as_dict()
        except Exception:
            args = {}
        payloads.append(
            {"tool_name": call.tool_name, "call_id": call.tool_call_id, "arguments": args}
        )
    return payloads


def queued_payload(payload: dict[str, Any], remaining: list[str]) -> dict[str, Any]:
    """The ``pending_tool_call`` payload with the ids still queued behind it."""
    return {**payload, "queued": list(remaining)}


def _pending_calls_from_segments(
    is_host: Callable[[str], bool], segments: list[dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    """Host-tool entries without an output in persisted assistant segments, by call id."""
    if not segments:
        return {}
    pending: dict[str, dict[str, Any]] = {}
    for segment in segments:
        if segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict) or "output" in tool:
                continue
            if not is_host(str(tool.get("name", ""))):
                continue
            pending[str(tool.get("id", ""))] = {
                "tool_name": str(tool.get("name", "")),
                "call_id": str(tool.get("id", "")),
                "arguments": tool.get("input") if isinstance(tool.get("input"), dict) else {},
            }
    return pending
