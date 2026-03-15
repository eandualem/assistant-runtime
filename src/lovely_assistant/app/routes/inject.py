"""Assistant inject & sessions endpoints — backbone/agent message delivery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from lovely_assistant.services.database.deps import DatabaseServiceDep
from lovely_assistant.services.database.repositories import InboxRepository

router = APIRouter(prefix="/assistant", tags=["assistant-inject"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class InjectRequest(BaseModel):
    from_agent: str = Field(..., alias="from", min_length=1)
    via: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(None, alias="sessionId")

    model_config = {"populate_by_name": True}


def _get_assistant_service(request: Request):
    """Retrieve AssistantService from app state (may be None during startup)."""
    return getattr(request.app.state, "assistant_service", None)


# ---------------------------------------------------------------------------
# POST /assistant/inject
# ---------------------------------------------------------------------------


@router.post("/inject", status_code=201)
async def inject_message(
    body: InjectRequest,
    request: Request,
    db: DatabaseServiceDep,
) -> dict[str, Any]:
    """Inject a message into a Jarvis session via the inbox.

    If session_id is provided and the session exists, tags the inbox item
    with that session and returns status "delivered". Otherwise stores
    the message without session tagging and returns "deferred".
    """
    service = _get_assistant_service(request)

    # Determine whether the target session exists
    session_found = False
    if body.session_id and service is not None:
        sessions = service.get_session_store()
        if sessions is not None:
            ctx = await sessions.get_context_if_exists_async(body.session_id)
            session_found = ctx is not None

    # Build context dict for the inbox item
    context: dict[str, Any] = {"via": body.via, "injected": True}
    if body.session_id and session_found:
        context["session_id"] = body.session_id

    status = "delivered" if session_found else "deferred"

    try:
        async with db.session_context() as session:
            repo = InboxRepository(session)
            row = await repo.create(
                from_agent=body.from_agent,
                message=body.message,
                severity="info",
                context=context,
            )
            inbox_id = row.id
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create inject inbox item", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to store injected message") from e

    return {"status": status, "inbox_id": inbox_id}


# ---------------------------------------------------------------------------
# GET /assistant/sessions
# ---------------------------------------------------------------------------


@router.get("/sessions")
async def list_sessions_backbone(request: Request) -> dict[str, Any]:
    """List active sessions in backbone-expected format.

    Returns {"sessions": [{"id": "...", "active": true, ...}]} — distinct
    from the frontend's GET /sessions which returns a flat list.
    """
    service = _get_assistant_service(request)
    if service is None:
        return {"sessions": []}

    sessions_store = service.get_session_store()
    if sessions_store is None:
        return {"sessions": []}

    raw = await sessions_store.list_sessions()

    backbone_sessions = []
    for s in raw:
        backbone_sessions.append(
            {
                "id": s.get("session_id", s.get("id")),
                "active": True,
                "title": s.get("title"),
                "turn_number": s.get("turn_number", 0),
            }
        )

    return {"sessions": backbone_sessions}
