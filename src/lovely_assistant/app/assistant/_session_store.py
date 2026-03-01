"""Session store with read-through DB persistence.

Manages per-session context (working memory, turn count, pending tool calls).
In-memory dict acts as hot cache; DB is the durable backing store when available.

Durability contract:
- When DB writes/reads succeed, session state is durable across restarts.
- When DB load fails after retries, the error propagates — callers get a clear
  error instead of silently losing conversation history.
- DB write failures (_persist_to_db) are still best-effort — the session continues
  in-memory. Write failures don't lose existing conversation state.
"""

from __future__ import annotations

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
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._db: DatabaseService | None = database_service
        self._session_ttl_hours = session_ttl_hours

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get or create session context (in-memory only — sync path)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "turn_number": 0,
                "working_memory": None,
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
            db_results = [
                {
                    "session_id": row.id,
                    "title": row.title,
                    "turn_number": row.turn_number,
                    "message_count": len(row.message_history) if row.message_history else 0,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

            # Overlay in-memory data on DB results — active sessions may have
            # newer message counts/titles not yet persisted to DB.
            for result in db_results:
                sid = result["session_id"]
                if sid in self._sessions:
                    ctx = self._sessions[sid]
                    result["message_count"] = len(ctx.get("message_history", []))
                    result["turn_number"] = ctx.get("turn_number", result["turn_number"])
                    if ctx.get("title"):
                        result["title"] = ctx["title"]

            return db_results

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
        Deserialization failures return the session with empty history (data is in DB
        but unreadable by the current pydantic-ai version) and log at ERROR level.
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
                raw_history = row.message_history if row.message_history else []
                try:
                    message_history = deserialize_messages(raw_history)
                except Exception as e:
                    logger.error(
                        "Session message history failed to deserialize — "
                        "returning session with empty history. "
                        "Data is still in DB and may be recoverable after a pydantic-ai upgrade.",
                        session_id=session_id,
                        stored_message_count=len(raw_history),
                        error_type=type(e).__name__,
                        error=str(e),
                    )
                    message_history = []

                return {
                    "turn_number": row.turn_number,
                    "working_memory": row.working_memory,
                    "message_history": message_history,
                    "title": row.title,
                }

        return await _load()

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
                "[SESSION] Failed to persist session to DB; state remains in-memory only",
                session_id=session_id,
                error=str(e),
            )
