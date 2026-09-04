"""Repair of host-tool calls that never received their result.

``pending_tool_call_id`` lives in memory only. After a reload, or when the
host never answered, a tool entry without an ``output`` would wait forever
and break the history; these helpers give it a synthetic output.
"""

from __future__ import annotations

import copy
from typing import Any

from loguru import logger

STALE_HOST_TOOL_OUTPUT = (
    "[Deferred host tool was not completed before the session was reloaded. "
    "The action did not complete.]"
)


def repair_stale_tool_segments(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """A copy of ``segments`` where every tool without output is marked stale.

    Returns ``(repaired_segments, repaired_tool_ids)``.
    """
    repaired_ids: list[str] = []
    updated = copy.deepcopy(segments)
    for segment in updated:
        if segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict):
                continue
            if "output" not in tool:
                tool["output"] = STALE_HOST_TOOL_OUTPUT
                repaired_ids.append(str(tool.get("id", "")))
    return updated, repaired_ids


def repair_stale_tools_in_context(
    ctx: dict[str, Any],
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
        updated_segments, repaired_ids = repair_stale_tool_segments(segments)
        if repaired_ids:
            record["segments"] = updated_segments
            repaired.append((msg_id, updated_segments))
            logger.debug(
                "[SESSION] Marked stale host tools",
                message_id=msg_id,
                repaired_tool_ids=repaired_ids,
            )
    return repaired
