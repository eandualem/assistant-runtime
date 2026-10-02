"""Settings queries; the caller owns the transaction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    UserSettingsORM,
)


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
        if not state:
            row = await self.get()
            if row is not None:
                return row
            result = await self._session.execute(
                insert(UserSettingsORM).values(id=self.SETTINGS_ID).returning(UserSettingsORM)
            )
            await self._session.flush()
            return result.scalar_one()

        stmt = pg_insert(UserSettingsORM).values(id=self.SETTINGS_ID, **state)
        result = await self._session.execute(
            stmt.on_conflict_do_update(
                index_elements=["id"],
                set_={
                    **{key: getattr(stmt.excluded, key) for key in state},
                    "updated_at": func.now(),
                },
            ).returning(UserSettingsORM)
        )
        await self._session.flush()
        return result.scalar_one()
