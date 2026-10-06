"""Sessions queries; the caller owns the transaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.base.cursors import ChangeCursor
from assistant_runtime.services.database.models import (
    MessageORM,
    PromptSnapshotORM,
    SessionORM,
    SteeringORM,
)

from ._ordering import _in_change_order


class SessionRepository:
    """CRUD operations for sessions. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        session_id: str,
        title: str | None = None,
        expires_at: datetime | None = None,
        owner_id: str | None = None,
        profile: str | None = None,
        subject: str | None = None,
    ) -> SessionORM:
        """Create a new session row."""
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": 0,
            "owner_id": owner_id,
            "profile": profile,
            "subject": subject,
        }
        if expires_at is not None:
            values["expires_at"] = expires_at

        result = await self._session.execute(
            insert(SessionORM).values(**values).returning(SessionORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def get(self, session_id: str) -> SessionORM | None:
        """Get a non-expired session by ID, or None."""
        result = await self._session.execute(
            select(SessionORM).where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        owner_id: str | None = None,
        updated_after: datetime | None = None,
    ) -> list[SessionORM]:
        """List non-expired sessions, most recently updated first.

        With ``owner_id``, only that principal's sessions are listed; with
        ``updated_after`` (timezone-aware), only those updated strictly later.
        """
        stmt = select(SessionORM).where(SessionORM.expires_at > func.now())
        if owner_id is not None:
            stmt = stmt.where(SessionORM.owner_id == owner_id)
        if updated_after is not None:
            stmt = stmt.where(SessionORM.updated_at > updated_after)
        result = await self._session.execute(
            stmt.order_by(SessionORM.updated_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def get_latest_by_telegram_chat_id(self, chat_id: str) -> SessionORM | None:
        """Return the newest Telegram-bound session for a chat, or None."""
        result = await self._session.execute(
            select(SessionORM)
            .where(
                SessionORM.telegram_chat_id == chat_id,
                SessionORM.expires_at > func.now(),
            )
            .order_by(SessionORM.telegram_bound_at.desc(), SessionORM.updated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def update(self, session_id: str, **fields) -> None:
        """Update specific fields on a session. No-op if session doesn't exist."""
        if not fields:
            return
        stmt = (
            update(SessionORM)
            .where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
            .values(**fields, updated_at=func.now())
        )
        await self._session.execute(stmt)
        await self._session.flush()

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        result = await self._session.execute(delete(SessionORM).where(SessionORM.id == session_id))
        await self._session.flush()
        return (result.rowcount or 0) > 0

    async def cleanup_expired(self) -> int:
        """Delete all expired sessions. Returns count of deleted rows."""
        result = await self._session.execute(
            delete(SessionORM).where(SessionORM.expires_at <= func.now())
        )
        await self._session.flush()
        return result.rowcount or 0

    async def upsert(
        self,
        session_id: str,
        *,
        title: str | None = None,
        turn_number: int = 0,
        working_memory: dict | None = None,
        telegram_chat_id: str | None = None,
        telegram_bound_at: datetime | None = None,
        expires_at: datetime | None = None,
        owner_id: str | None = None,
        pending_action: dict | None = None,
        profile: str | None = None,
        subject: str | None = None,
    ) -> None:
        """Atomic INSERT ... ON CONFLICT DO UPDATE.

        Eliminates the race condition in check-then-insert patterns. The
        owner is written on insert only: reassignment goes through
        ``update()``, so a stale state save can never restore an owner an
        administrator cleared or changed. The profile and subject binding
        is written on insert only too: it never changes.
        """
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": turn_number,
            "working_memory": working_memory,
            "telegram_chat_id": telegram_chat_id,
            "telegram_bound_at": telegram_bound_at,
            "owner_id": owner_id,
            "pending_action": pending_action,
            "profile": profile,
            "subject": subject,
        }
        if expires_at is not None:
            values["expires_at"] = expires_at

        stmt = pg_insert(SessionORM).values(**values)

        update_fields: dict[str, object] = {
            "title": stmt.excluded.title,
            "turn_number": stmt.excluded.turn_number,
            "working_memory": stmt.excluded.working_memory,
            "telegram_chat_id": stmt.excluded.telegram_chat_id,
            "telegram_bound_at": stmt.excluded.telegram_bound_at,
            "pending_action": stmt.excluded.pending_action,
            "updated_at": func.now(),
        }
        if expires_at is not None:
            update_fields["expires_at"] = stmt.excluded.expires_at

        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_fields)
        await self._session.execute(stmt)
        await self._session.flush()


class MessageRepository:
    """CRUD operations for conversation messages."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        message_id: str,
        session_id: str,
        parent_id: str | None,
        role: str,
        message_type: str,
        content: str,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        prompt: dict[str, Any] | None = None,
        model_messages: list[dict[str, Any]] | None = None,
    ) -> MessageORM:
        result = await self._session.execute(
            insert(MessageORM)
            .values(
                id=message_id,
                session_id=session_id,
                parent_id=parent_id,
                role=role,
                message_type=message_type,
                content=content,
                segments=segments,
                usage=usage,
                prompt=prompt,
                model_messages=model_messages,
            )
            .returning(MessageORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def get(self, message_id: str) -> MessageORM | None:
        result = await self._session.execute(select(MessageORM).where(MessageORM.id == message_id))
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: str) -> list[MessageORM]:
        result = await self._session.execute(
            select(MessageORM)
            .where(MessageORM.session_id == session_id)
            .order_by(MessageORM.created_at.asc())
        )
        return list(result.scalars().all())

    async def list_changes(
        self, session_id: str, *, after: ChangeCursor | None, limit: int
    ) -> list[MessageORM]:
        """The session's messages changed after ``after`` (all without it), oldest change first."""
        query = _in_change_order(
            select(MessageORM).where(MessageORM.session_id == session_id), MessageORM, after
        )
        result = await self._session.execute(query.limit(limit))
        return list(result.scalars().all())

    async def count_by_sessions(self, session_ids: list[str]) -> dict[str, int]:
        if not session_ids:
            return {}
        result = await self._session.execute(
            select(MessageORM.session_id, func.count(MessageORM.id))
            .where(MessageORM.session_id.in_(session_ids))
            .group_by(MessageORM.session_id)
        )
        return {session_id: int(count) for session_id, count in result.all()}

    async def count_created(self, start: datetime, end: datetime) -> int:
        """Messages created in ``[start, end)``, across every session."""
        result = await self._session.execute(
            select(func.count())
            .select_from(MessageORM)
            .where(MessageORM.created_at >= start, MessageORM.created_at < end)
        )
        return int(result.scalar_one())

    async def update(
        self,
        message_id: str,
        *,
        content: str | None = None,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
        prompt: dict[str, Any] | None = None,
        model_messages: list[dict[str, Any]] | None = None,
    ) -> None:
        fields: dict[str, Any] = {}
        if content is not None:
            fields["content"] = content
        if segments is not None:
            fields["segments"] = segments
        if usage is not None:
            fields["usage"] = usage
        if prompt is not None:
            fields["prompt"] = prompt
        if model_messages is not None:
            fields["model_messages"] = model_messages
        if not fields:
            return
        await self._session.execute(
            update(MessageORM)
            .where(MessageORM.id == message_id)
            .values(**fields, updated_at=func.now())
        )
        await self._session.flush()


class PromptSnapshotRepository:
    """Stable prompt texts by hash. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def put(self, snapshot_hash: str, content: str) -> None:
        """Store the text unless its hash is already there."""
        await self._session.execute(
            pg_insert(PromptSnapshotORM)
            .values(hash=snapshot_hash, content=content)
            .on_conflict_do_nothing(index_elements=["hash"])
        )
        await self._session.flush()

    async def get(self, snapshot_hash: str) -> str | None:
        result = await self._session.execute(
            select(PromptSnapshotORM.content).where(PromptSnapshotORM.hash == snapshot_hash)
        )
        return result.scalar_one_or_none()


class SteeringRepository:
    """CRUD operations for steering records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        steering_id: str,
        session_id: str,
        content: str,
        status: str,
        profile: str | None = None,
        delivered_at: datetime | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> SteeringORM:
        result = await self._session.execute(
            insert(SteeringORM)
            .values(
                id=steering_id,
                session_id=session_id,
                profile=profile,
                content=content,
                status=status,
                delivered_at=delivered_at,
                attachments=attachments,
            )
            .returning(SteeringORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list_by_session(self, session_id: str) -> list[SteeringORM]:
        result = await self._session.execute(
            select(SteeringORM)
            .where(SteeringORM.session_id == session_id)
            .order_by(SteeringORM.created_at.asc(), SteeringORM.id.asc())
        )
        return list(result.scalars().all())

    async def mark_status(
        self,
        steering_ids: list[str],
        *,
        status: str,
        delivered_at: datetime | None,
    ) -> None:
        if not steering_ids:
            return

        await self._session.execute(
            update(SteeringORM)
            .where(SteeringORM.id.in_(steering_ids))
            .values(status=status, delivered_at=delivered_at)
        )
        await self._session.flush()
