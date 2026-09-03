"""Session management endpoints — /sessions/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from assistant_runtime.app.assistant._serialization import (
    merge_display_messages,
    tree_messages_to_tree,
)
from assistant_runtime.app.assistant.deps import AssistantServiceDep

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

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
        "has_pending_tool_call": bool(ctx.get("pending_tool_call_id")),
        "message_count": ctx.get("message_count", 0),
    }


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    service: AssistantServiceDep,
    leaf_id: str | None = None,
) -> list[dict]:
    """Get the root-to-leaf display path for a session."""
    sessions = service.get_session_store()
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        path = await sessions.get_message_path(session_id, leaf_id=leaf_id)
        steering = await sessions.get_display_steering(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return merge_display_messages(path, steering)


@router.get("/sessions/{session_id}/tree")
async def get_session_tree(session_id: str, service: AssistantServiceDep) -> list[dict]:
    """Get all messages in a session with parent pointers."""
    sessions = service.get_session_store()
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        messages = await sessions.get_tree_messages(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return tree_messages_to_tree(messages)


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


@router.post("/sessions/{session_id}/repair")
async def repair_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Repair stale frontend tools stuck in a session.

    Clears any in-memory pending tool call state and marks unresolved
    frontend tools in message segments as stale (both in-memory and DB).
    """
    sessions = service.get_session_store()
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        report = await sessions.repair_stale_frontend_tools(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return report


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
            from assistant_runtime.services.database.repositories import TraceRepository

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
