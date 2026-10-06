"""Commit ordering for cursor-based record feeds, and change-cursor queries."""

from typing import Any

from sqlalchemy import ColumnElement, Select, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from assistant_runtime.base.cursors import ChangeCursor


async def _in_commit_order(session: AsyncSession, feed: str) -> None:
    """Hold the feed's lock until commit, so its ids become visible in increasing order.

    Identity values are handed out when a row is inserted, not when it
    commits; without this a reader continuing ``after`` a later id could
    skip an earlier one that commits afterwards.
    """
    key = f"assistant_runtime:{feed}"
    await session.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0))))


def _changed_after(model: Any, cursor: ChangeCursor) -> ColumnElement[bool]:
    """Rows after ``cursor`` in ``(updated_at, id)`` order; a bare time keeps later times only."""
    if cursor.id is None:
        return model.updated_at > cursor.updated_at
    return tuple_(model.updated_at, model.id) > tuple_(cursor.updated_at, cursor.id)


def _in_change_order(query: Select, model: Any, cursor: ChangeCursor | None) -> Select:
    """``query`` filtered after ``cursor`` (when given), oldest change first."""
    if cursor is not None:
        query = query.where(_changed_after(model, cursor))
    return query.order_by(model.updated_at.asc(), model.id.asc())
