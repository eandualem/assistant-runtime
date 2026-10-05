"""Record counts — how many records each store created in a time range (administration).

A host's tooling compares these with its own index to see what it is missing.
A store that is not running contributes 0; voice calls carry no timestamps yet
and are not counted.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query, Request

from assistant_runtime.app.access.deps import AdminDep

router = APIRouter(prefix="/counts", tags=["counts"])


def _aware(value: datetime) -> datetime:
    """A timezone-less value is taken as UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


@router.get("")
async def get_counts(
    request: Request,
    admin: AdminDep,
    start: datetime = Query(alias="from"),
    end: datetime = Query(alias="to"),
) -> dict:
    """Records created in ``[from, to)``: events, tasks, actions, agent and conversation messages.

    ``from`` and ``to`` are ISO 8601 (UTC when they have no timezone); ``422``
    unless ``from`` is before ``to``.
    """
    start, end = _aware(start), _aware(end)
    if start >= end:
        raise HTTPException(status_code=422, detail="'from' must be before 'to'")
    state = request.app.state
    event_log = getattr(state, "event_log_service", None)
    tasks = getattr(state, "task_service", None)
    actions = getattr(state, "action_service", None)
    assistant = getattr(state, "assistant_service", None)
    sessions = assistant.get_session_store() if assistant is not None else None
    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "events": await event_log.count(start, end) if event_log is not None else 0,
        "tasks": await tasks.count(start, end) if tasks is not None else 0,
        "actions": await actions.count(start, end) if actions is not None else 0,
        "agent_messages": await tasks.count_messages(start, end) if tasks is not None else 0,
        "messages": await sessions.count_messages(start, end) if sessions is not None else 0,
    }
