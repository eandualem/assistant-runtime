"""Database repositories — query layer over ORM models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import case, delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from lovely_assistant.services.database.models import (
    InboxItemORM,
    SessionORM,
    TraceORM,
    UserSettingsORM,
)

# Severity ordering for unsurfaced inbox queries (highest priority first).
_SEVERITY_ORDER = {"urgent": 0, "action_needed": 1, "info": 2}


class SessionRepository:
    """CRUD operations for sessions. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        session_id: str,
        title: str | None = None,
        expires_at: datetime | None = None,
    ) -> SessionORM:
        """Create a new session row."""
        row = SessionORM(id=session_id, title=title, turn_number=0)
        if expires_at is not None:
            row.expires_at = expires_at
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, session_id: str) -> SessionORM | None:
        """Get a non-expired session by ID, or None."""
        result = await self._session.execute(
            select(SessionORM).where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[SessionORM]:
        """List non-expired sessions ordered by most recently updated."""
        result = await self._session.execute(
            select(SessionORM)
            .where(SessionORM.expires_at > func.now())
            .order_by(SessionORM.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

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

    async def exists(self, session_id: str) -> bool:
        """Check if a non-expired session exists."""
        result = await self._session.execute(
            select(SessionORM.id).where(
                SessionORM.id == session_id,
                SessionORM.expires_at > func.now(),
            )
        )
        return result.scalar_one_or_none() is not None

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
        message_history: list | None = None,
        working_memory: dict | None = None,
        pending_tool_call: dict | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        """Atomic INSERT ... ON CONFLICT DO UPDATE.

        Eliminates the race condition in check-then-insert patterns.
        """
        values: dict[str, object] = {
            "id": session_id,
            "title": title,
            "turn_number": turn_number,
            "message_history": message_history or [],
            "working_memory": working_memory,
            "pending_tool_call": pending_tool_call,
        }
        if expires_at is not None:
            values["expires_at"] = expires_at

        stmt = pg_insert(SessionORM).values(**values)

        update_fields: dict[str, object] = {
            "title": stmt.excluded.title,
            "turn_number": stmt.excluded.turn_number,
            "message_history": stmt.excluded.message_history,
            "working_memory": stmt.excluded.working_memory,
            "pending_tool_call": stmt.excluded.pending_tool_call,
            "updated_at": func.now(),
        }
        if expires_at is not None:
            update_fields["expires_at"] = stmt.excluded.expires_at

        stmt = stmt.on_conflict_do_update(index_elements=["id"], set_=update_fields)
        await self._session.execute(stmt)
        await self._session.flush()


class TraceRepository:
    """CRUD operations for debug traces. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        trace_id: str,
        session_id: str,
        events: list[dict[str, Any]],
        *,
        user_message: str | None = None,
        is_continuation: bool = False,
        duration_ms: float | None = None,
        screenshot: str | None = None,
    ) -> TraceORM:
        """Create a new trace row."""
        row = TraceORM(
            id=trace_id,
            session_id=session_id,
            events=events,
            user_message=user_message,
            is_continuation=is_continuation,
            duration_ms=duration_ms,
            screenshot=screenshot,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_by_session(
        self, session_id: str, limit: int = 50, offset: int = 0
    ) -> list[TraceORM]:
        """List traces for a session, ordered by creation time."""
        result = await self._session.execute(
            select(TraceORM)
            .where(TraceORM.session_id == session_id)
            .order_by(TraceORM.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())


class SettingsRepository:
    """CRUD operations for user settings. Uses flush() — caller owns commit."""

    SETTINGS_ID: str = "default"

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self) -> UserSettingsORM | None:
        """Get the settings row, or None."""
        result = await self._session.execute(
            select(UserSettingsORM).where(UserSettingsORM.id == self.SETTINGS_ID)
        )
        return result.scalar_one_or_none()

    async def save(self, state: dict[str, Any]) -> UserSettingsORM:
        """Upsert settings — create or update the single settings row."""
        row = await self.get()
        if row is None:
            row = UserSettingsORM(id=self.SETTINGS_ID)
            self._session.add(row)
        for key, value in state.items():
            setattr(row, key, value)
        await self._session.flush()
        return row


class InboxRepository:
    """CRUD operations for inbox items. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        from_agent: str,
        message: str,
        severity: str = "info",
        context: dict[str, Any] | None = None,
    ) -> InboxItemORM:
        """Create a new inbox item."""
        row = InboxItemORM(
            from_agent=from_agent,
            message=message,
            severity=severity,
            context=context,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def list_unsurfaced(self, limit: int = 20) -> list[InboxItemORM]:
        """List unsurfaced items ordered by severity (urgent first) then created_at desc."""
        severity_sort = case(
            _SEVERITY_ORDER,
            value=InboxItemORM.severity,
            else_=99,
        )
        result = await self._session.execute(
            select(InboxItemORM)
            .where(InboxItemORM.surfaced.is_(False))
            .order_by(severity_sort, InboxItemORM.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_surfaced(self, item_id: str) -> InboxItemORM | None:
        """Mark an item as surfaced. Returns the item or None if not found."""
        result = await self._session.execute(select(InboxItemORM).where(InboxItemORM.id == item_id))
        row = result.scalar_one_or_none()
        if row is not None:
            row.surfaced = True
            await self._session.flush()
        return row

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[InboxItemORM]:
        """List all items ordered by created_at desc."""
        result = await self._session.execute(
            select(InboxItemORM)
            .order_by(InboxItemORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())
