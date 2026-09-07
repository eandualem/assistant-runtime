"""Inbox endpoints — a note for the assistant, delivered like any other message.

``POST /inbox`` goes through the ingress (delivered into the most recent
session, queued when there is none); the other endpoints inspect the
queue, which lives in Postgres.
"""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import require_admin
from assistant_runtime.app.ingress.deps import IngressServiceDep
from assistant_runtime.services.database.deps import get_database_service
from assistant_runtime.services.database.repositories import InboxRepository

router = APIRouter(prefix="/inbox", tags=["inbox"], dependencies=[Depends(require_admin)])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InboxItemCreate(BaseModel):
    from_agent: str = Field(..., alias="from", min_length=1)
    message: str = Field(..., min_length=1)
    severity: Literal["info", "action_needed", "urgent"] = "info"
    context: dict[str, Any] | None = None

    model_config = {"populate_by_name": True}


class InboxItemResponse(BaseModel):
    id: str
    from_agent: str
    message: str
    severity: str
    context: dict[str, Any] | None
    surfaced: bool
    created_at: str


def _row_to_response(row) -> dict:
    """Convert an InboxItemORM row to a response dict."""
    return {
        "id": row.id,
        "from_agent": row.from_agent,
        "message": row.message,
        "severity": row.severity,
        "context": row.context,
        "surfaced": row.surfaced,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=201)
async def create_inbox_item(body: InboxItemCreate, ingress: IngressServiceDep) -> dict:
    """Leave a note for the assistant; delivered into the most recent session or queued."""
    session_id = (body.context or {}).get("session_id")
    try:
        return await ingress.deliver(
            from_agent=body.from_agent,
            via=(body.context or {}).get("via", "inbox"),
            message=body.message,
            session_id=session_id if isinstance(session_id, str) else None,
            severity=body.severity,
        )
    except Exception as e:
        logger.error("Failed to deliver inbox item", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to deliver the note") from e


@router.get("")
async def list_inbox_items(
    request: Request,
    surfaced: bool | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """List inbox items. Filter by surfaced status if provided."""
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = InboxRepository(session)
            if surfaced is False:
                rows = await repo.list_unsurfaced(limit=limit)
            else:
                rows = await repo.list_all(limit=limit, offset=offset)
            return [_row_to_response(row) for row in rows]
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to list inbox items", error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e


@router.patch("/{item_id}/surfaced")
async def mark_item_surfaced(item_id: str, request: Request) -> dict:
    """Mark an inbox item as surfaced."""
    db = get_database_service(request)

    try:
        async with db.session_context() as session:
            repo = InboxRepository(session)
            row = await repo.mark_surfaced(item_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Inbox item not found")
            return _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to mark item surfaced", item_id=item_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update inbox item") from e
