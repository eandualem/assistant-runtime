"""Session management endpoints — /sessions/*."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lovely_assistant.app.assistant.deps import AssistantServiceDep

router = APIRouter()


@router.get("/sessions")
async def list_sessions(
    service: AssistantServiceDep,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List sessions (metadata only — no full message history)."""
    sessions = service._sessions
    if sessions is None:
        return []

    return await sessions.list_sessions(limit=limit, offset=offset)


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Get session info (turn count, pending tool call, message count).

    Checks in-memory cache first, then tries DB. Returns 404 if not found in either.
    """
    sessions = service._sessions
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check in-memory first
    if sessions.has_session(session_id):
        ctx = sessions.get_context(session_id)
    else:
        # Try loading from DB
        loaded = await sessions._load_session_from_db(session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found")
        # Cache in memory
        sessions._sessions[session_id] = loaded
        ctx = loaded

    return {
        "session_id": session_id,
        "turn_number": ctx.get("turn_number", 0),
        "has_pending_tool_call": ctx.get("pending_tool_call") is not None,
        "message_count": len(ctx.get("message_history", [])),
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Delete a session and all its context."""
    sessions = service._sessions
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if exists in memory or DB
    if not sessions.has_session(session_id):
        # Try DB
        loaded = await sessions._load_session_from_db(session_id)
        if loaded is None:
            raise HTTPException(status_code=404, detail="Session not found")

    await sessions.delete_session(session_id)
    return {"session_id": session_id, "deleted": True}
