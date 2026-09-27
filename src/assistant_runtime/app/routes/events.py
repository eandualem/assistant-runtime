"""Event endpoints — record, list and advance events (see ``app/event_log``).

An administrator (the host) writes events: what other systems reported
(inbound), and notices for the owner (outbound).
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from assistant_runtime.app.access.deps import AdminDep
from assistant_runtime.app.event_log.deps import EventLogServiceDep
from assistant_runtime.app.event_log.exceptions import (
    EventConflictError,
    EventLogError,
    EventNotFoundError,
)

router = APIRouter(prefix="/events", tags=["events"])

_STATUS = {EventNotFoundError: 404, EventConflictError: 409}


class EventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=200)
    """With ``source``, the idempotency key: the same pair is stored once."""
    direction: Literal["inbound", "outbound"]
    source: str = Field(min_length=1, max_length=64)
    agent: str | None = Field(default=None, min_length=1, max_length=128)
    kind: str = Field(min_length=1, max_length=64)
    severity: Literal["info", "warning", "critical"] = "info"
    summary: str = Field(default="", max_length=4000)
    payload: dict[str, Any] | None = None
    target_session_id: str | None = Field(default=None, min_length=1, max_length=64)
    """An inbound event naming a session is steered into it; others are only stored."""
    history: bool = False
    """Imported from before: stored, never steered, left out by ``news_only``."""
    occurred_at: AwareDatetime | None = None


class NoticeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["delivered", "heard"]


def _http_error(exc: EventLogError) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 422), detail=str(exc))


@router.post("", status_code=201)
async def record_event(
    body: EventRequest, response: Response, events: EventLogServiceDep, admin: AdminDep
) -> dict:
    """Store an event (``201``), or return the stored one for a repeated key (``200``)."""
    try:
        record, created = await events.record(**body.model_dump())
    except EventLogError as e:
        raise _http_error(e) from e
    if not created:
        response.status_code = 200
    return record.to_dict()


@router.get("")
async def list_events(
    events: EventLogServiceDep,
    admin: AdminDep,
    after: int = Query(0, ge=0),
    direction: Literal["inbound", "outbound"] | None = None,
    source: str | None = None,
    agent: str | None = None,
    kind: str | None = None,
    news_only: bool = False,
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    """Events after ``after`` in arrival order; continue from ``next_after``."""
    records = await events.list(
        after=after,
        limit=limit,
        direction=direction,
        source=source,
        agent=agent,
        kind=kind,
        news_only=news_only,
    )
    return {
        "events": [r.to_dict() for r in records],
        "next_after": records[-1].id if records else after,
    }


@router.get("/{event_id}")
async def get_event(event_id: int, events: EventLogServiceDep, admin: AdminDep) -> dict:
    try:
        return (await events.get(event_id)).to_dict()
    except EventLogError as e:
        raise _http_error(e) from e


@router.patch("/{event_id}")
async def update_notice(
    event_id: int, body: NoticeUpdate, events: EventLogServiceDep, admin: AdminDep
) -> dict:
    """Move an outbound notice forward: ``pending`` → ``delivered`` → ``heard``."""
    try:
        return (await events.update_notice(event_id, body.status)).to_dict()
    except EventLogError as e:
        raise _http_error(e) from e
