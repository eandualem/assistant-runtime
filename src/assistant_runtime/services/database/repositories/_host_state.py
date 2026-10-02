"""Host state queries; the caller owns the transaction."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    HostStateORM,
)


class HostStateRepository:
    """Versioned host values. Every write is one conditional statement, so none is lost.

    A deleted key is kept as a tombstone with its version, so a version is
    never reused and a stale write cannot match a value created later.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, namespace: str, key: str) -> HostStateORM | None:
        result = await self._session.execute(
            select(HostStateORM).where(
                HostStateORM.namespace == namespace,
                HostStateORM.key == key,
                HostStateORM.deleted.is_(False),
            )
        )
        return result.scalar_one_or_none()

    async def list(self, namespace: str, after: str | None, limit: int) -> list[HostStateORM]:
        query = select(HostStateORM).where(
            HostStateORM.namespace == namespace, HostStateORM.deleted.is_(False)
        )
        if after is not None:
            query = query.where(HostStateORM.key > after)
        result = await self._session.execute(query.order_by(HostStateORM.key).limit(limit))
        return list(result.scalars().all())

    async def put(
        self, namespace: str, key: str, value: Any, by: str, expected_version: int | None
    ) -> HostStateORM | None:
        """Write the value; None when ``expected_version`` does not match what is stored.

        ``expected_version`` 0 means no value may be stored (a tombstone is
        revived with its next version); ``None`` writes whatever is there.
        """
        model = HostStateORM
        if expected_version is None or expected_version == 0:
            statement = pg_insert(model).values(
                namespace=namespace, key=key, value=value, version=1, updated_by=by
            )
            statement = statement.on_conflict_do_update(
                index_elements=["namespace", "key"],
                set_={
                    "value": statement.excluded.value,
                    "version": model.version + 1,
                    "deleted": False,
                    "updated_by": statement.excluded.updated_by,
                    "updated_at": func.now(),
                },
                where=model.deleted.is_(True) if expected_version == 0 else None,
            )
        else:
            statement = (
                update(model)
                .where(
                    model.namespace == namespace,
                    model.key == key,
                    model.version == expected_version,
                    model.deleted.is_(False),
                )
                .values(
                    value=value, version=model.version + 1, updated_by=by, updated_at=func.now()
                )
            )
        result = await self._session.execute(statement.returning(model))
        await self._session.flush()
        return result.scalar_one_or_none()

    async def delete(self, namespace: str, key: str, by: str, expected_version: int | None) -> bool:
        """Leave a tombstone at the next version; False when nothing matched."""
        model = HostStateORM
        statement = update(model).where(
            model.namespace == namespace, model.key == key, model.deleted.is_(False)
        )
        if expected_version is not None:
            statement = statement.where(model.version == expected_version)
        result = await self._session.execute(
            statement.values(
                value=None,
                deleted=True,
                version=model.version + 1,
                updated_by=by,
                updated_at=func.now(),
            ).returning(model.key)
        )
        await self._session.flush()
        return result.scalar_one_or_none() is not None
