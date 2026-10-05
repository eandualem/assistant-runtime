"""Events queries; the caller owns the transaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.base.cursors import ChangeCursor
from assistant_runtime.services.database.models import (
    EventORM,
)

from ._ordering import _in_change_order, _in_commit_order


class EventRepository:
    """Event records, in arrival order. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create_if_new(self, **values: Any) -> tuple[EventORM, bool]:
        """Insert the event, or return the stored one with the same (source, event_id)."""
        await _in_commit_order(self._session, "events")
        result = await self._session.execute(
            pg_insert(EventORM)
            .values(**values)
            .on_conflict_do_nothing(constraint="uq_events_source_event_id")
            .returning(EventORM)
        )
        row = result.scalar_one_or_none()
        await self._session.flush()
        if row is not None:
            return row, True
        existing = await self._session.execute(
            select(EventORM).where(
                EventORM.source == values["source"], EventORM.event_id == values["event_id"]
            )
        )
        return existing.scalar_one(), False

    async def get(self, event_id: int, *, lock: bool = False) -> EventORM | None:
        query = select(EventORM).where(EventORM.id == event_id)
        result = await self._session.execute(query.with_for_update() if lock else query)
        return result.scalar_one_or_none()

    async def update(self, event_id: int, **fields: Any) -> EventORM | None:
        result = await self._session.execute(
            update(EventORM)
            .where(EventORM.id == event_id)
            .values({**fields, "updated_at": func.now()})
            .returning(EventORM)
        )
        await self._session.flush()
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        after: int,
        limit: int,
        updated_after: ChangeCursor | None = None,
        **filters: Any,
    ) -> list[EventORM]:
        """Events after ``after`` in id order; ``history=False`` leaves out imported history.

        With ``updated_after``, the events changed after that cursor, oldest change first.
        """
        query = select(EventORM).where(EventORM.id > after)
        for name, value in filters.items():
            if value is not None:
                query = query.where(getattr(EventORM, name) == value)
        if updated_after is not None:
            query = _in_change_order(query, EventORM, updated_after)
        else:
            query = query.order_by(EventORM.id)
        result = await self._session.execute(query.limit(limit))
        return list(result.scalars().all())

    async def count_created(self, start: datetime, end: datetime) -> int:
        """Events created in ``[start, end)``."""
        result = await self._session.execute(
            select(func.count())
            .select_from(EventORM)
            .where(EventORM.created_at >= start, EventORM.created_at < end)
        )
        return int(result.scalar_one())
