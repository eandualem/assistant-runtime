"""Resolution of host-tool calls that will never receive their result.

A tool entry without an ``output`` waits for a continuation. When that can
no longer arrive (the session was reloaded without a matching persisted
pending action, or an operator repaired it), the entry gets a synthetic
output with ``outcome: "interrupted"`` and a ``status`` saying why, so the
history stays consistent without asserting that the external action failed.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from loguru import logger

ActionStatus = Literal["completed", "failed", "cancelled", "superseded", "unknown"]
"""How a host action was resolved, recorded as ``status`` on its tool entry."""

STALE_HOST_TOOL_OUTPUT = (
    "[Deferred host tool was not completed before the session was reloaded. "
    "Whether the action took effect is unknown.]"
)


def resolve_tool_entry(tool: dict[str, Any], *, output: Any, status: ActionStatus) -> None:
    """Give an unanswered tool entry a synthetic result, in place."""
    tool["output"] = output
    tool["outcome"] = "interrupted"
    tool["status"] = status


def repair_stale_tool_segments(
    segments: list[dict[str, Any]],
    *,
    keep: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[str]]:
    """A copy of ``segments`` where every tool without output is marked ``unknown``.

    Tool call ids in ``keep`` (the persisted pending action) stay unanswered.
    Returns ``(repaired_segments, repaired_tool_ids)``.
    """
    repaired_ids: list[str] = []
    updated = copy.deepcopy(segments)
    for segment in updated:
        if not isinstance(segment, dict) or segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict) or "output" in tool:
                continue
            if str(tool.get("id", "")) in keep:
                continue
            resolve_tool_entry(tool, output=STALE_HOST_TOOL_OUTPUT, status="unknown")
            repaired_ids.append(str(tool.get("id", "")))
    return updated, repaired_ids


def repair_stale_tools_in_context(
    ctx: dict[str, Any],
    *,
    keep: frozenset[str] = frozenset(),
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Mark stale tools on every assistant message in ``ctx`` (in place).

    Returns ``(message_id, repaired_segments)`` for each message that changed,
    so the caller can persist them.
    """
    repaired: list[tuple[str, list[dict[str, Any]]]] = []
    for msg_id, record in ctx["message_index"].items():
        if record.get("role") != "assistant":
            continue
        segments = record.get("segments")
        if not segments:
            continue
        updated_segments, repaired_ids = repair_stale_tool_segments(segments, keep=keep)
        if repaired_ids:
            record["segments"] = updated_segments
            repaired.append((msg_id, updated_segments))
            logger.debug(
                "[SESSION] Marked stale host tools",
                message_id=msg_id,
                repaired_tool_ids=repaired_ids,
            )
    return repaired


def find_tool_entry(
    segments: list[dict[str, Any]] | None, tool_call_id: str
) -> dict[str, Any] | None:
    """The tool entry with ``tool_call_id`` in ``segments``, or None."""
    for segment in segments or []:
        if not isinstance(segment, dict) or segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if isinstance(tool, dict) and str(tool.get("id", "")) == tool_call_id:
                return tool
    return None
