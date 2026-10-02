"""Commit ordering for cursor-based record feeds."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def _in_commit_order(session: AsyncSession, feed: str) -> None:
    """Hold the feed's lock until commit, so its ids become visible in increasing order.

    Identity values are handed out when a row is inserted, not when it
    commits; without this a reader continuing ``after`` a later id could
    skip an earlier one that commits afterwards.
    """
    key = f"assistant_runtime:{feed}"
    await session.execute(select(func.pg_advisory_xact_lock(func.hashtextextended(key, 0))))
