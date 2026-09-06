"""Session management endpoints — /sessions/*."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import AdminDep, PrincipalDep
from assistant_runtime.app.assistant._serialization import (
    merge_display_messages,
    tree_messages_to_tree,
)
from assistant_runtime.app.assistant._stale_tools import find_tool_entry
from assistant_runtime.app.assistant.deps import AssistantServiceDep
from assistant_runtime.principal import Principal, can_access_session

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

router = APIRouter()


async def _get_session_context(session_id: str, sessions: Any, principal: Principal) -> dict:
    """Look up session context (memory, then DB); 404 when absent, 403 when not the caller's."""
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")

    ctx = await sessions.get_context_if_exists_async(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Session not found")
    _authorize(ctx, principal, session_id)
    return ctx


def _authorize(ctx: dict, principal: Principal, session_id: str) -> None:
    if not can_access_session(principal, ctx.get("owner_id")):
        raise HTTPException(
            status_code=403, detail=f"Session '{session_id}' belongs to another principal"
        )


class OwnerUpdate(BaseModel):
    owner_id: str | None = Field(default=None, max_length=128)


@router.get("/sessions")
async def list_sessions(
    service: AssistantServiceDep,
    principal: PrincipalDep,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List the caller's sessions (every session for an administrator); metadata only."""
    sessions = service.get_session_store()
    if sessions is None:
        return []

    return await sessions.list_sessions(
        limit=limit, offset=offset, owner_id=None if principal.is_admin else principal.id
    )


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str, service: AssistantServiceDep, principal: PrincipalDep
) -> dict:
    """Get session info (turn count, pending tool call, message count).

    Checks in-memory cache first, then tries DB. Returns 404 if not found in either.
    """
    ctx = await _get_session_context(session_id, service.get_session_store(), principal)

    return {
        "session_id": session_id,
        "owner_id": ctx.get("owner_id"),
        "turn_number": ctx.get("turn_number", 0),
        "has_pending_tool_call": bool(ctx.get("pending_tool_call_id")),
        "pending_action": _pending_action(ctx),
        "message_count": ctx.get("message_count", 0),
    }


def _pending_action(ctx: dict) -> dict | None:
    """The pending host action with its arguments, so a host can perform it without the transcript."""
    pending_id = ctx.get("pending_tool_call_id")
    if not pending_id:
        return None
    message_id = ctx.get("pending_assistant_message_id")
    record = ctx["message_index"].get(message_id) if message_id else None
    entry = find_tool_entry(record.get("segments"), str(pending_id)) if record else None
    arguments = entry.get("input") if entry and isinstance(entry.get("input"), dict) else {}
    batch = [str(c) for c in (ctx.get("pending_tool_batch") or [pending_id])]
    return {
        "tool_call_id": pending_id,
        "tool_name": ctx.get("pending_tool_name"),
        "assistant_message_id": message_id,
        "arguments": arguments,
        # Host calls from the same response still to be handed over, in order.
        "queued": [c for c in batch if c != pending_id and _unanswered(record, c)],
    }


def _unanswered(record: dict | None, tool_call_id: str) -> bool:
    entry = find_tool_entry(record.get("segments"), tool_call_id) if record else None
    return entry is not None and "output" not in entry


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    service: AssistantServiceDep,
    principal: PrincipalDep,
    leaf_id: str | None = None,
) -> list[dict]:
    """Get the root-to-leaf display path for a session."""
    sessions = service.get_session_store()
    await _get_session_context(session_id, sessions, principal)
    try:
        path = await sessions.get_message_path(session_id, leaf_id=leaf_id)
        steering = await sessions.get_display_steering(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return merge_display_messages(path, steering)


@router.get("/sessions/{session_id}/tree")
async def get_session_tree(
    session_id: str, service: AssistantServiceDep, principal: PrincipalDep
) -> list[dict]:
    """Get all messages in a session with parent pointers."""
    sessions = service.get_session_store()
    await _get_session_context(session_id, sessions, principal)
    try:
        messages = await sessions.get_tree_messages(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return tree_messages_to_tree(messages)


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, service: AssistantServiceDep, principal: PrincipalDep
) -> dict:
    """Delete a session and all its context."""
    sessions = service.get_session_store()
    await _get_session_context(session_id, sessions, principal)
    await sessions.delete_session(session_id)
    return {"session_id": session_id, "deleted": True}


@router.patch("/sessions/{session_id}/owner")
async def set_session_owner(
    session_id: str, body: OwnerUpdate, service: AssistantServiceDep, admin: AdminDep
) -> dict:
    """Assign a session to a principal (administration; null makes it unowned)."""
    sessions = service.get_session_store()
    if sessions is None:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        await sessions.set_owner(session_id, body.owner_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return {"session_id": session_id, "owner_id": body.owner_id}


@router.post("/sessions/{session_id}/repair")
async def repair_session(
    session_id: str, service: AssistantServiceDep, principal: PrincipalDep
) -> dict:
    """Resolve host tools stuck in a session as ``unknown``.

    Clears the pending host action (in memory and on the row) and gives
    every tool without a result a synthetic one, so the session can continue.
    """
    sessions = service.get_session_store()
    await _get_session_context(session_id, sessions, principal)

    try:
        report = await sessions.repair_stale_host_tools(session_id)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return report


@router.get("/sessions/{session_id}/traces")
async def get_session_traces(
    session_id: str,
    service: AssistantServiceDep,
    principal: PrincipalDep,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get debug traces for a session, ordered by creation time."""
    await _get_session_context(session_id, service.get_session_store(), principal)
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
