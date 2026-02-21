"""Session store with read-through DB persistence.

Manages per-session context (working memory, turn count, pending tool calls).
In-memory dict acts as hot cache; DB is the durable backing store.
Falls back to pure in-memory mode when DB is unavailable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.messages import ModelMessage
from sqlalchemy.exc import DisconnectionError, InterfaceError, OperationalError

from lovely_assistant.app.assistant._serialization import (
    deserialize_messages,
    serialize_messages,
)
from lovely_assistant.base.resilience import retry_with_backoff

_DB_RETRYABLE_EXCEPTIONS = (
    OperationalError,
    DisconnectionError,
    InterfaceError,
    ConnectionError,
    TimeoutError,
)

if TYPE_CHECKING:
    from lovely_assistant.services.database.interface import DatabaseService


_MAX_MEMORY_SESSIONS = 200


class SessionStore:
    """Session state storage with optional DB persistence."""

    def __init__(
        self,
        database_service: DatabaseService | None = None,
        session_ttl_hours: int = 24,
        pending_tool_call_timeout_minutes: int = 10,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._db: DatabaseService | None = database_service
        self._session_ttl_hours = session_ttl_hours
        self._pending_timeout_minutes = pending_tool_call_timeout_minutes

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

        # Cache miss — evict if memory is full before adding a new entry
        self._evict_if_needed()

        # Try loading from DB
        if self._db is not None:
            loaded = await self._load_session_from_db(session_id)
            if loaded is not None:
                self._sessions[session_id] = loaded
                return loaded

        # Create new
        return self.get_context(session_id)

    async def get_context_if_exists_async(self, session_id: str) -> dict[str, Any] | None:
        """Get session context if it exists (memory or DB). Returns None if not found."""
        if session_id in self._sessions:
            return self._sessions[session_id]

        if self._db is not None:
            loaded = await self._load_session_from_db(session_id)
            if loaded is not None:
                self._sessions[session_id] = loaded
                return loaded

        return None

    def get_history(self, session_id: str) -> list[ModelMessage]:
        """Get message history for a session."""
        ctx = self.get_context(session_id)
        history = ctx.get("message_history", [])
        logger.debug(
            "Retrieved session history",
            session_id=session_id,
            message_count=len(history),
        )
        return history

    def save_history(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save message history for a session (in-memory only)."""
        ctx = self.get_context(session_id)
        ctx["message_history"] = messages

    async def save_history_async(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save message history and persist to DB."""
        ctx = self.get_context(session_id)
        ctx["message_history"] = messages
        logger.debug(
            "[SESSION] Saving session history",
            session_id=session_id,
            message_count=len(messages),
        )

        # Auto-title from first user message if no title set
        if not ctx.get("title") and messages:
            from pydantic_ai.messages import ModelRequest, UserPromptPart

            for msg in messages:
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                            text = part.content.strip()
                            ctx["title"] = text[:50] + ("..." if len(text) > 50 else "")
                            break
                if ctx.get("title"):
                    break

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
        *,
        rejected_call_ids: list[str] | None = None,
    ) -> None:
        """Store a pending frontend tool call for continuation."""
        ctx = self.get_context(session_id)
        pending: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "created_at": time.time(),
        }
        if rejected_call_ids:
            pending["rejected_call_ids"] = rejected_call_ids
        ctx["pending_tool_call"] = pending

    def clear_pending_tool_call(self, session_id: str) -> dict[str, Any] | None:
        """Clear and return the pending tool call, if any.

        Returns None if no pending call exists or if the pending call has expired.
        """
        ctx = self.get_context(session_id)
        pending = ctx.get("pending_tool_call")
        ctx["pending_tool_call"] = None
        if pending is None:
            return None

        # Check expiry (backward-compat: old entries may lack created_at)
        created_at = pending.get("created_at")
        if created_at is not None:
            elapsed_minutes = (time.time() - created_at) / 60
            if elapsed_minutes > self._pending_timeout_minutes:
                logger.warning(
                    "Pending tool call expired",
                    session_id=session_id,
                    tool_name=pending.get("tool_name"),
                    elapsed_minutes=round(elapsed_minutes, 1),
                    timeout_minutes=self._pending_timeout_minutes,
                )
                return None

        return pending

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists in memory."""
        return session_id in self._sessions

    def session_count(self) -> int:
        """Number of active sessions in memory."""
        return len(self._sessions)

    async def delete_session(self, session_id: str) -> None:
        """Remove session from DB first, then memory."""
        if self._db is not None:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SessionRepository,
                )

                repo = SessionRepository(db_session)
                await repo.delete(session_id)

        self._sessions.pop(session_id, None)

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List sessions from DB (metadata only — no full history)."""
        if self._db is None:
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

    def _evict_if_needed(self) -> None:
        """Evict oldest half of in-memory sessions if above threshold."""
        if len(self._sessions) <= _MAX_MEMORY_SESSIONS:
            return
        # dict preserves insertion order in Python 3.7+ — oldest entries are first
        evict_count = len(self._sessions) // 2
        keys_to_evict = list(self._sessions.keys())[:evict_count]
        for key in keys_to_evict:
            del self._sessions[key]
        logger.info(
            "[SESSION] Evicted in-memory sessions",
            evicted=evict_count,
            remaining=len(self._sessions),
        )

    async def cleanup_expired(self) -> int:
        """Delete expired sessions from DB. Returns count deleted."""
        if self._db is None:
            return 0

        async with self._db.session_context() as db_session:
            from lovely_assistant.services.database.repositories import (
                SessionRepository,
            )

            repo = SessionRepository(db_session)
            return await repo.cleanup_expired()

    async def _load_session_from_db(self, session_id: str) -> dict[str, Any] | None:
        """Load a session from DB into the in-memory format.

        Retries on transient DB errors (OperationalError, DisconnectionError, etc.).
        """
        if self._db is None:
            return None

        @retry_with_backoff(
            max_attempts=4,
            retry_on=_DB_RETRYABLE_EXCEPTIONS,
            name="load_session_from_db",
        )
        async def _load() -> dict[str, Any] | None:
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

        try:
            return await _load()
        except Exception as e:
            logger.warning(
                "Failed to load session from DB",
                session_id=session_id,
                error=str(e),
            )
            return None

    @staticmethod
    def _jsonb_safe(value: object) -> object:
        """Convert Pydantic models to dicts for JSONB storage."""
        if value is None:
            return None
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            return value.model_dump()
        return value

    async def _persist_to_db(self, session_id: str) -> None:
        """Persist current in-memory session state to DB via atomic upsert.

        Retries on transient DB errors (OperationalError, DisconnectionError, etc.).
        """
        if self._db is None:
            return

        ctx = self._sessions.get(session_id)
        if ctx is None:
            return

        # Serialize outside the retry loop — deterministic and doesn't need DB
        messages = ctx.get("message_history", [])
        serialized_history = serialize_messages(messages) if messages else []
        working_memory = self._jsonb_safe(ctx.get("working_memory"))
        pending_tool_call = self._jsonb_safe(ctx.get("pending_tool_call"))
        expires_at = datetime.now(UTC) + timedelta(hours=self._session_ttl_hours)

        @retry_with_backoff(
            max_attempts=4,
            retry_on=_DB_RETRYABLE_EXCEPTIONS,
            name="persist_to_db",
        )
        async def _persist() -> None:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SessionRepository,
                )

                repo = SessionRepository(db_session)
                await repo.upsert(
                    session_id,
                    title=ctx.get("title"),
                    turn_number=ctx.get("turn_number", 0),
                    message_history=serialized_history,
                    working_memory=working_memory,
                    pending_tool_call=pending_tool_call,
                    expires_at=expires_at,
                )

                logger.debug(
                    "[SESSION] Persisted session to DB",
                    session_id=session_id,
                    message_count=len(messages),
                )

        try:
            await _persist()
        except Exception as e:
            logger.exception(
                "[SESSION] Failed to persist session to DB",
                session_id=session_id,
                error=str(e),
            )
