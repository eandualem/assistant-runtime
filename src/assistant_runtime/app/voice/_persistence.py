"""Optional transcript/usage checkpoints, independent of backend message rows."""

from datetime import UTC, datetime

from loguru import logger
from sqlalchemy import func, select, tuple_
from sqlalchemy.dialects.postgresql import insert

from assistant_runtime.base.cursors import ChangeCursor
from assistant_runtime.services.database.interface import DatabaseService
from assistant_runtime.services.database.models import VoiceCallORM


def _snapshot(row: VoiceCallORM) -> dict:
    """The stored snapshot, with the row's change time (the one cursors order by)."""
    return {**row.snapshot, "updated_at": row.updated_at.isoformat()}


class VoicePersistence:
    def __init__(self, database: DatabaseService | None):
        self.database = database

    @property
    def available(self) -> bool:
        return self.database is not None and self.database.healthy

    async def save(self, record: dict) -> bool:
        if not self.available:
            return False
        try:
            values = {"id": record["call_id"], "session_id": record["session_id"]}
            if isinstance(record.get("created_at"), int | float):  # Unix seconds
                values["created_at"] = datetime.fromtimestamp(record["created_at"], UTC)
            async with self.database.session_context() as db:
                query = insert(VoiceCallORM).values(**values, snapshot=record)
                await db.execute(
                    query.on_conflict_do_update(
                        index_elements=[VoiceCallORM.id],
                        set_={"snapshot": query.excluded.snapshot, "updated_at": func.now()},
                    )
                )
            return True
        except Exception:
            # Provider events can contain conversation text. Do not log payloads
            # or SQL parameter strings when the optional database is unavailable.
            logger.warning("Voice checkpoint unavailable", call_id=record["call_id"])
            return False

    async def load(self, call_id: str) -> dict | None:
        if not self.available:
            return None
        # A failed lookup raises: "not found" would be wrong while the database is lost.
        async with self.database.session_context() as db:
            row = await db.get(VoiceCallORM, call_id)
            return _snapshot(row) if row is not None else None

    async def list_changes(
        self, *, after: ChangeCursor | None, limit: int, owner_id: str | None
    ) -> list[dict]:
        """Snapshots changed after ``after``, oldest change first; ``owner_id`` keeps theirs."""
        query = select(VoiceCallORM)
        if owner_id is not None:
            query = query.where(VoiceCallORM.snapshot["owner_id"].astext == owner_id)
        if after is not None and after.id is None:
            query = query.where(VoiceCallORM.updated_at > after.updated_at)
        elif after is not None:
            key = tuple_(VoiceCallORM.updated_at, VoiceCallORM.id)
            query = query.where(key > tuple_(after.updated_at, after.id))
        query = query.order_by(VoiceCallORM.updated_at, VoiceCallORM.id).limit(limit)
        async with self.database.session_context() as db:
            rows = (await db.execute(query)).scalars().all()
            return [_snapshot(row) for row in rows]

    async def count_created(self, start: datetime, end: datetime) -> int:
        query = (
            select(func.count())
            .select_from(VoiceCallORM)
            .where(VoiceCallORM.created_at >= start, VoiceCallORM.created_at < end)
        )
        async with self.database.session_context() as db:
            return int((await db.execute(query)).scalar_one())
