"""The session tree in Postgres: every read and write the session store makes.

The store keeps sessions in memory and calls one method here per change;
this module is the only place that knows the repositories. Connection
errors on a cold load are retried; everything else propagates so the
store never pretends a write happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import DisconnectionError, InterfaceError, OperationalError

from assistant_runtime.app.assistant._serialization import MessageRecord, SteeringRecord
from assistant_runtime.base.resilience import retry_with_backoff
from assistant_runtime.services.database.repositories import (
    MessageRepository,
    SessionRepository,
    SteeringRepository,
)

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

_DB_RETRYABLE_EXCEPTIONS = (
    OperationalError,
    DisconnectionError,
    InterfaceError,
    ConnectionError,
    TimeoutError,
)


@dataclass(frozen=True)
class LoadedSession:
    """A session row with its messages and steering, as stored."""

    turn_number: int
    working_memory: Any
    title: str | None
    owner_id: str | None
    telegram_chat_id: str | None
    telegram_bound_at: datetime | None
    messages: list[MessageRecord]
    steering: list[SteeringRecord]


class SessionPersistence:
    """Session, message and steering rows behind one database service."""

    def __init__(self, database_service: DatabaseService, session_ttl_hours: int) -> None:
        self._db = database_service
        self._session_ttl_hours = session_ttl_hours

    def _expires_at(self) -> datetime:
        return datetime.now(UTC) + timedelta(hours=self._session_ttl_hours)

    async def ensure_session(
        self, session_id: str, title: str | None, owner_id: str | None = None
    ) -> None:
        """Create the session row unless it exists."""
        async with self._db.session_context() as db_session:
            repo = SessionRepository(db_session)
            if await repo.get(session_id) is None:
                await repo.create(
                    session_id=session_id,
                    title=title,
                    expires_at=self._expires_at(),
                    owner_id=owner_id,
                )

    async def set_owner(self, session_id: str, owner_id: str | None) -> None:
        async with self._db.session_context() as db_session:
            await SessionRepository(db_session).update(session_id, owner_id=owner_id)

    async def create_message(self, record: MessageRecord) -> None:
        async with self._db.session_context() as db_session:
            await MessageRepository(db_session).create(
                message_id=record["id"],
                session_id=record["session_id"],
                parent_id=record["parent_id"],
                role=record["role"],
                message_type=record["message_type"],
                content=record["content"],
                segments=record["segments"],
                usage=record["usage"],
            )

    async def update_message(
        self,
        message_id: str,
        *,
        content: str | None = None,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> None:
        async with self._db.session_context() as db_session:
            await MessageRepository(db_session).update(
                message_id, content=content, segments=segments, usage=usage
            )

    async def update_segments(self, repaired: list[tuple[str, list[dict[str, Any]]]]) -> None:
        """Write repaired segments for several messages in one transaction."""
        async with self._db.session_context() as db_session:
            repo = MessageRepository(db_session)
            for message_id, segments in repaired:
                await repo.update(message_id, segments=segments)

    async def create_steering(self, record: SteeringRecord) -> None:
        async with self._db.session_context() as db_session:
            await SteeringRepository(db_session).create(
                steering_id=record["id"],
                session_id=record["session_id"],
                content=record["content"],
                status=record["status"],
                delivered_at=record["delivered_at"],
            )

    async def mark_steering(
        self, steering_ids: list[str], *, status: str, delivered_at: datetime
    ) -> None:
        async with self._db.session_context() as db_session:
            await SteeringRepository(db_session).mark_status(
                steering_ids, status=status, delivered_at=delivered_at
            )

    async def save_state(self, session_id: str, ctx: dict[str, Any]) -> None:
        """Persist the non-message session metadata."""
        async with self._db.session_context() as db_session:
            await SessionRepository(db_session).upsert(
                session_id,
                title=ctx.get("title"),
                turn_number=ctx.get("turn_number", 0),
                working_memory=ctx.get("working_memory"),
                telegram_chat_id=ctx.get("telegram_chat_id"),
                telegram_bound_at=ctx.get("telegram_bound_at"),
                expires_at=self._expires_at(),
                owner_id=ctx.get("owner_id"),
            )

    async def session_for_telegram_chat(self, chat_id: str) -> str | None:
        async with self._db.session_context() as db_session:
            row = await SessionRepository(db_session).get_latest_by_telegram_chat_id(chat_id)
            return row.id if row is not None else None

    async def delete(self, session_id: str) -> None:
        async with self._db.session_context() as db_session:
            await SessionRepository(db_session).delete(session_id)

    async def list_sessions(
        self, limit: int, offset: int, *, owner_id: str | None = None
    ) -> list[dict[str, Any]]:
        async with self._db.session_context() as db_session:
            rows = await SessionRepository(db_session).list_all(
                limit=limit, offset=offset, owner_id=owner_id
            )
            counts = await MessageRepository(db_session).count_by_sessions([row.id for row in rows])
            return [
                {
                    "session_id": row.id,
                    "owner_id": row.owner_id,
                    "title": row.title,
                    "turn_number": row.turn_number,
                    "message_count": counts.get(row.id, 0),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    async def cleanup_expired(self) -> int:
        async with self._db.session_context() as db_session:
            return await SessionRepository(db_session).cleanup_expired()

    async def load(self, session_id: str) -> LoadedSession | None:
        """Everything stored for a session, or None when there is no row."""

        @retry_with_backoff(
            max_attempts=3, min_wait=0.1, max_wait=1.0, retry_on=_DB_RETRYABLE_EXCEPTIONS
        )
        async def _load() -> LoadedSession | None:
            async with self._db.session_context() as db_session:
                row = await SessionRepository(db_session).get(session_id)
                if row is None:
                    return None
                messages = await MessageRepository(db_session).list_by_session(session_id)
                steering = await SteeringRepository(db_session).list_by_session(session_id)
                return LoadedSession(
                    turn_number=row.turn_number,
                    working_memory=row.working_memory,
                    title=row.title,
                    owner_id=row.owner_id,
                    telegram_chat_id=row.telegram_chat_id,
                    telegram_bound_at=row.telegram_bound_at,
                    messages=[
                        {
                            "id": m.id,
                            "session_id": m.session_id,
                            "parent_id": m.parent_id,
                            "role": m.role,
                            "message_type": m.message_type,
                            "content": m.content,
                            "segments": m.segments,
                            "usage": m.usage,
                            "created_at": m.created_at,
                        }
                        for m in messages
                    ],
                    steering=[
                        {
                            "id": s.id,
                            "session_id": s.session_id,
                            "content": s.content,
                            "status": s.status,
                            "created_at": s.created_at,
                            "delivered_at": s.delivered_at,
                        }
                        for s in steering
                    ],
                )

        return await _load()
