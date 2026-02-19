"""Session management endpoints — /sessions/*."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lovely_assistant.app.assistant.deps import AssistantServiceDep

router = APIRouter()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, service: AssistantServiceDep) -> dict:
    """Get session info (turn count, pending tool call, message count)."""
    sessions = service._sessions
    if sessions is None or not sessions.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    ctx = sessions.get_context(session_id)
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
    if sessions is None or not sessions.has_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove session from internal store
    sessions._sessions.pop(session_id, None)
    return {"session_id": session_id, "deleted": True}
