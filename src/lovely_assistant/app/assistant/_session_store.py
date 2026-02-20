"""Session store with read-through DB persistence.

Manages per-session context (working memory, turn count, pending tool calls).
In-memory dict acts as hot cache; DB is the durable backing store.
Falls back to pure in-memory mode when DB is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.messages import ModelMessage

from lovely_assistant.app.assistant._serialization import (
    deserialize_messages,
    serialize_messages,
)

if TYPE_CHECKING:
    from lovely_assistant.services.database.interface import DatabaseService


class SessionStore:
    """Session state storage with optional DB persistence."""

    def __init__(self, database_service: DatabaseService | None = None) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._db: DatabaseService | None = database_service

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get or create session context (in-memory only — sync path)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "turn_number": 0,
                "working_memory": None,
                "pending_tool_call": None,
                "message_history": [],
            }
        return self._sessions[session_id]

    async def get_context_async(self, session_id: str) -> dict[str, Any]:
        """Get or create session context, loading from DB on cache miss."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        # Try loading from DB
        if self._db is not None and self._db._healthy:
            loaded = await self._load_session_from_db(session_id)
            if loaded is not None:
                self._sessions[session_id] = loaded
                return loaded

        # Create new
        return self.get_context(session_id)

    def get_history(self, session_id: str) -> list[ModelMessage]:
        """Get message history for a session."""
        ctx = self.get_context(session_id)
        return ctx.get("message_history", [])

    def save_history(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save message history for a session (in-memory only)."""
        ctx = self.get_context(session_id)
        ctx["message_history"] = messages

    async def save_history_async(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save message history and persist to DB."""
        ctx = self.get_context(session_id)
        ctx["message_history"] = messages

        # Persist full session state to DB
        await self._persist_to_db(session_id)

    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn number."""
        ctx = self.get_context(session_id)
        ctx["turn_number"] = ctx.get("turn_number", 0) + 1
        return ctx["turn_number"]

    def set_pending_tool_call(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        """Store a pending frontend tool call for continuation."""
        ctx = self.get_context(session_id)
        ctx["pending_tool_call"] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }

    def clear_pending_tool_call(self, session_id: str) -> dict[str, str] | None:
        """Clear and return the pending tool call, if any."""
        ctx = self.get_context(session_id)
        pending = ctx.get("pending_tool_call")
        ctx["pending_tool_call"] = None
        return pending

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists in memory."""
        return session_id in self._sessions

    def session_count(self) -> int:
        """Number of active sessions in memory."""
        return len(self._sessions)

    async def delete_session(self, session_id: str) -> None:
        """Remove session from memory and DB."""
        self._sessions.pop(session_id, None)

        if self._db is not None and self._db._healthy:
            try:
                async with self._db.session_context() as db_session:
                    from lovely_assistant.services.database.repositories import (
                        SessionRepository,
                    )

                    repo = SessionRepository(db_session)
                    await repo.delete(session_id)
            except Exception as e:
                logger.warning(
                    "Failed to delete session from DB", session_id=session_id, error=str(e)
                )

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List sessions from DB (metadata only — no full history)."""
        if self._db is None or not self._db._healthy:
            # Fallback: return in-memory sessions
            sessions = []
            for sid, ctx in self._sessions.items():
                sessions.append(
                    {
                        "session_id": sid,
                        "title": ctx.get("title"),
                        "turn_number": ctx.get("turn_number", 0),
                        "message_count": len(ctx.get("message_history", [])),
                        "created_at": None,
                    }
                )
            return sessions[offset : offset + limit]

        try:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SessionRepository,
                )

                repo = SessionRepository(db_session)
                rows = await repo.list_all(limit=limit, offset=offset)
                return [
                    {
                        "session_id": row.id,
                        "title": row.title,
                        "turn_number": row.turn_number,
                        "message_count": len(row.message_history) if row.message_history else 0,
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.warning("Failed to list sessions from DB, using in-memory", error=str(e))
            return [
                {
                    "session_id": sid,
                    "title": ctx.get("title"),
                    "turn_number": ctx.get("turn_number", 0),
                    "message_count": len(ctx.get("message_history", [])),
                    "created_at": None,
                }
                for sid, ctx in list(self._sessions.items())[offset : offset + limit]
            ]

    async def _load_session_from_db(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from DB into the in-memory format."""
        if self._db is None:
            return None

        try:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SessionRepository,
                )

                repo = SessionRepository(db_session)
                row = await repo.get(session_id)
                if row is None:
                    return None

                # Deserialize JSONB message history back to typed ModelMessage list
                message_history = deserialize_messages(
                    row.message_history if row.message_history else []
                )

                return {
                    "turn_number": row.turn_number,
                    "working_memory": row.working_memory,
                    "pending_tool_call": row.pending_tool_call,
                    "message_history": message_history,
                    "title": row.title,
                }
        except Exception as e:
            logger.warning(
                "Failed to load session from DB",
                session_id=session_id,
                error=str(e),
            )
            return None

    async def _persist_to_db(self, session_id: str) -> None:
        """Persist current in-memory session state to DB."""
        if self._db is None or not self._db._healthy:
            return

        ctx = self._sessions.get(session_id)
        if ctx is None:
            return

        try:
            # Serialize message history to JSON-compatible format
            messages = ctx.get("message_history", [])
            serialized_history = serialize_messages(messages) if messages else []

            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SessionRepository,
                )

                repo = SessionRepository(db_session)
                existing = await repo.get(session_id)

                if existing is None:
                    # Create new row
                    row = await repo.create(session_id, title=ctx.get("title"))
                    row.turn_number = ctx.get("turn_number", 0)
                    row.message_history = serialized_history
                    row.working_memory = ctx.get("working_memory")
                    row.pending_tool_call = ctx.get("pending_tool_call")
                    await db_session.flush()
                else:
                    # Update existing row
                    await repo.update(
                        session_id,
                        turn_number=ctx.get("turn_number", 0),
                        message_history=serialized_history,
                        working_memory=ctx.get("working_memory"),
                        pending_tool_call=ctx.get("pending_tool_call"),
                        title=ctx.get("title"),
                    )
        except Exception as e:
            logger.warning(
                "Failed to persist session to DB",
                session_id=session_id,
                error=str(e),
            )
