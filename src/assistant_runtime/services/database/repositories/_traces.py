"""Traces queries; the caller owns the transaction."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.services.database.models import (
    TraceORM,
)


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
    ) -> TraceORM:
        """Create a new trace row."""
        result = await self._session.execute(
            insert(TraceORM)
            .values(
                id=trace_id,
                session_id=session_id,
                events=events,
                user_message=user_message,
                is_continuation=is_continuation,
                duration_ms=duration_ms,
            )
            .returning(TraceORM)
        )
        await self._session.flush()
        return result.scalar_one()

    async def delete_older_than(self, cutoff: datetime) -> int:
        """Delete traces created before ``cutoff``. Returns the number of rows removed."""
        result = await self._session.execute(delete(TraceORM).where(TraceORM.created_at < cutoff))
        await self._session.flush()
        return result.rowcount or 0

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
