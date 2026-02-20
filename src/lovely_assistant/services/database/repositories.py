"""Database repositories — query layer over ORM models."""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from lovely_assistant.services.database.models import SessionORM, UserSettingsORM


class SessionRepository:
    """CRUD operations for sessions. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, session_id: str, title: str | None = None) -> SessionORM:
        """Create a new session row."""
        row = SessionORM(id=session_id, title=title, turn_number=0)
        self._session.add(row)
        await self._session.flush()
        return row

    async def get(self, session_id: str) -> SessionORM | None:
        """Get a session by ID, or None."""
        result = await self._session.execute(select(SessionORM).where(SessionORM.id == session_id))
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[SessionORM]:
        """List sessions ordered by most recently updated."""
        result = await self._session.execute(
            select(SessionORM).order_by(SessionORM.updated_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def update(self, session_id: str, **fields) -> None:
        """Update specific fields on a session. No-op if session doesn't exist."""
        row = await self.get(session_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
        await self._session.flush()

    async def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if a row was deleted."""
        result = await self._session.execute(delete(SessionORM).where(SessionORM.id == session_id))
        await self._session.flush()
        return (result.rowcount or 0) > 0

    async def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        result = await self._session.execute(
            select(SessionORM.id).where(SessionORM.id == session_id)
        )
        return result.scalar_one_or_none() is not None


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
