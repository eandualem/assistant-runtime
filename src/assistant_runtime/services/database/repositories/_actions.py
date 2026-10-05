"""Actions queries; the caller owns the transaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    ActionConfirmationORM,
    ActionORM,
)

from ._ordering import _in_commit_order


class ActionRepository:
    """Action records and their confirmations. Uses flush() — caller owns commit."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **values: Any) -> ActionORM:
        result = await self._session.execute(
            insert(ActionORM).values(**values).returning(ActionORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def get(self, action_id: int, *, lock: bool = False) -> ActionORM | None:
        query = select(ActionORM).where(ActionORM.id == action_id)
        result = await self._session.execute(query.with_for_update() if lock else query)
        return result.scalar_one_or_none()

    async def update(self, action_id: int, **fields: Any) -> ActionORM:
        result = await self._session.execute(
            update(ActionORM).where(ActionORM.id == action_id).values(**fields).returning(ActionORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list(
        self, *, limit: int, before: int | None = None, **filters: Any
    ) -> list[ActionORM]:
        """Newest first (by id), filtered by whatever is given; ``before`` lists lower ids only."""
        query = select(ActionORM)
        if before is not None:
            query = query.where(ActionORM.id < before)
        for name, value in filters.items():
            if value is not None:
                query = query.where(getattr(ActionORM, name) == value)
        result = await self._session.execute(query.order_by(ActionORM.id.desc()).limit(limit))
        return list(result.scalars().all())

    async def count_created(self, start: datetime, end: datetime) -> int:
        """Actions created in ``[start, end)``."""
        result = await self._session.execute(
            select(func.count())
            .select_from(ActionORM)
            .where(ActionORM.created_at >= start, ActionORM.created_at < end)
        )
        return int(result.scalar_one())

    async def add_confirmation(self, **values: Any) -> ActionConfirmationORM | None:
        """Insert a confirmation; None when its id is already taken."""
        await _in_commit_order(self._session, "action_confirmations")
        result = await self._session.execute(
            pg_insert(ActionConfirmationORM)
            .values(**values)
            .on_conflict_do_nothing(index_elements=["id"])
            .returning(ActionConfirmationORM)
        )
        await self._session.flush()
        return result.scalar_one_or_none()

    async def get_confirmation(
        self, confirmation_id: str, *, lock: bool = False
    ) -> ActionConfirmationORM | None:
        query = select(ActionConfirmationORM).where(ActionConfirmationORM.id == confirmation_id)
        result = await self._session.execute(query.with_for_update() if lock else query)
        return result.scalar_one_or_none()

    async def update_confirmation(
        self, confirmation_id: str, **fields: Any
    ) -> ActionConfirmationORM:
        result = await self._session.execute(
            update(ActionConfirmationORM)
            .where(ActionConfirmationORM.id == confirmation_id)
            .values(**fields)
            .returning(ActionConfirmationORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def list_confirmations(
        self,
        *,
        after: int,
        limit: int,
        signed: bool | None = None,
        reconciled: bool | None = None,
        **filters: Any,
    ) -> list[ActionConfirmationORM]:
        """Confirmations after ``after`` in insertion order."""
        model = ActionConfirmationORM
        query = select(model).where(model.seq > after)
        for name, value in filters.items():
            if value is not None:
                query = query.where(getattr(model, name) == value)
        if signed is not None:
            query = query.where(
                model.key_epoch.is_not(None) if signed else model.key_epoch.is_(None)
            )
        if reconciled is not None:
            query = query.where(
                model.reconciled.is_not(None) if reconciled else model.reconciled.is_(None)
            )
        result = await self._session.execute(query.order_by(model.seq).limit(limit))
        return list(result.scalars().all())
