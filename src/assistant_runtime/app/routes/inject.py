"""Message ingress endpoints — other systems reach the assistant here."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import require_admin
from assistant_runtime.app.ingress.deps import IngressServiceDep

router = APIRouter(
    prefix="/assistant", tags=["assistant-ingress"], dependencies=[Depends(require_admin)]
)


class InjectRequest(BaseModel):
    from_agent: str = Field(..., alias="from", min_length=1)
    via: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(None, alias="sessionId")
    telegram_chat_id: str | None = Field(None, alias="telegramChatId")

    model_config = {"populate_by_name": True}


@router.post("/inject", status_code=201)
async def inject_message(body: InjectRequest, ingress: IngressServiceDep) -> dict[str, Any]:
    """Deliver a message into a session (the named one, the one bound to the
    Telegram chat, or the most recent), or queue it until a session exists."""
    try:
        return await ingress.deliver(
            from_agent=body.from_agent,
            via=body.via,
            message=body.message,
            session_id=body.session_id,
            telegram_chat_id=body.telegram_chat_id,
        )
    except Exception as e:
        logger.error("Ingress delivery failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to deliver the message") from e


@router.get("/sessions")
async def list_sessions_for_peers(request: Request) -> dict[str, Any]:
    """List active sessions as ``{"sessions": [{"id", "active", "title", "turn_number"}]}``.

    The shape other systems expect; the host-facing list is ``GET /sessions``.
    """
    service = getattr(request.app.state, "assistant_service", None)
    store = service.get_session_store() if service is not None else None
    if store is None:
        return {"sessions": []}
    raw = await store.list_sessions()
    return {
        "sessions": [
            {
                "id": s.get("session_id", s.get("id")),
                "active": True,
                "title": s.get("title"),
                "turn_number": s.get("turn_number", 0),
            }
            for s in raw
        ]
    }
