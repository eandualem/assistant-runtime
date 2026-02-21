"""Inbox endpoints — agents push messages for Jarvis to surface contextually."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from lovely_assistant.services.database.interface import DatabaseService
from lovely_assistant.services.database.repositories import InboxRepository

router = APIRouter(prefix="/inbox", tags=["inbox"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def _get_db(request: Request) -> DatabaseService:
    """Retrieve DatabaseService from app state (independent of assistant module)."""
    db: DatabaseService | None = getattr(request.app.state, "database_service", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


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
async def create_inbox_item(body: InboxItemCreate, request: Request) -> dict:
    """Create a new inbox item from an agent."""
    db = await _get_db(request)

    try:
        async with db.session_context() as session:
            repo = InboxRepository(session)
            row = await repo.create(
                from_agent=body.from_agent,
                message=body.message,
                severity=body.severity,
                context=body.context,
            )
            result = _row_to_response(row)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create inbox item", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create inbox item") from e

    return result


@router.get("")
async def list_inbox_items(
    request: Request,
    surfaced: bool | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[dict]:
    """List inbox items. Filter by surfaced status if provided."""
    db = await _get_db(request)

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
        logger.warning("Failed to list inbox items", error=str(e))
        return []


@router.patch("/{item_id}/surfaced")
async def mark_item_surfaced(item_id: str, request: Request) -> dict:
    """Mark an inbox item as surfaced."""
    db = await _get_db(request)

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
