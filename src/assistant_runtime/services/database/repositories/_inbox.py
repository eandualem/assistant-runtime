"""Inbox queries; the caller owns the transaction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import case, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    InboxItemORM,
)

_SEVERITY_ORDER = {"urgent": 0, "action_needed": 1, "info": 2}


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
        result = await self._session.execute(
            insert(InboxItemORM)
            .values(
                from_agent=from_agent,
                message=message,
                severity=severity,
                context=context,
            )
            .returning(InboxItemORM)
        )
        await self._session.flush()
        return result.scalar_one()

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
        result = await self._session.execute(
            update(InboxItemORM)
            .where(InboxItemORM.id == item_id)
            .values(surfaced=True)
            .returning(InboxItemORM)
        )
        row = result.scalar_one_or_none()
        if row is not None:
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
