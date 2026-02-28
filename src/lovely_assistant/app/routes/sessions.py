"""Session management endpoints — /sessions/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from lovely_assistant.app.assistant._serialization import messages_to_display_format
from lovely_assistant.app.assistant.deps import AssistantServiceDep

if TYPE_CHECKING:
    from lovely_assistant.services.database.interface import DatabaseService

router = APIRouter()


async def _get_session_context(session_id: str, sessions: Any) -> dict:
    """Look up session context: memory first, then DB fallback, then 404."""
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    ctx = await sessions.get_context_if_exists_async(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return ctx


@router.get("/sessions")
async def list_sessions(
    service: AssistantServiceDep,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List sessions (metadata only — no full message history)."""
    sessions = service.get_session_store()
    if sessions is None:
        return []

    return await sessions.list_sessions(limit=limit, offset=offset)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Get session info (turn count, pending tool call, message count).

    Checks in-memory cache first, then tries DB. Returns 404 if not found in either.
    """
    ctx = await _get_session_context(session_id, service.get_session_store())

    return {
        "session_id": session_id,
        "turn_number": ctx.get("turn_number", 0),
        "message_count": len(ctx.get("message_history", [])),
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, service: AssistantServiceDep) -> list[dict]:
    """Get display-ready message history for a session."""
    ctx = await _get_session_context(session_id, service.get_session_store())
    return messages_to_display_format(ctx.get("message_history", []))


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Delete a session and all its context."""
    sessions = service.get_session_store()
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Verify existence (memory or DB) before deleting
    ctx = await sessions.get_context_if_exists_async(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Session not found")

    await sessions.delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


@router.get("/sessions/{session_id}/traces")
async def get_session_traces(
    session_id: str,
    service: AssistantServiceDep,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get debug traces for a session, ordered by creation time."""
    db: DatabaseService | None = service.get_database_service()
    if db is None:
        return []

    try:
        async with db.session_context() as db_session:
            from lovely_assistant.services.database.repositories import TraceRepository

            repo = TraceRepository(db_session)
            rows = await repo.list_by_session(session_id, limit=limit, offset=offset)
            return [
                {
                    "id": row.id,
                    "session_id": row.session_id,
                    "events": row.events,
                    "user_message": row.user_message,
                    "is_continuation": row.is_continuation,
                    "duration_ms": row.duration_ms,
                    "screenshot": row.screenshot,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]
    except Exception as e:
        logger.error("Failed to fetch traces from DB", session_id=session_id, error=str(e))
        raise HTTPException(status_code=503, detail="Database unavailable") from e
