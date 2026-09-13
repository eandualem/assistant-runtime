"""Optional transcript/usage checkpoints, independent of backend message rows."""

from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from assistant_runtime.services.database.interface import DatabaseService
from assistant_runtime.services.database.models import VoiceCallORM


class VoicePersistence:
    def __init__(self, database: DatabaseService | None):
        self.database = database

    async def save(self, record: dict) -> bool:
        if self.database is None or not self.database.healthy:
            return False
        try:
            async with self.database.session_context() as db:
                query = insert(VoiceCallORM).values(
                    id=record["call_id"], session_id=record["session_id"], snapshot=record
                )
                await db.execute(
                    query.on_conflict_do_update(
                        index_elements=[VoiceCallORM.id], set_={"snapshot": query.excluded.snapshot}
                    )
                )
            return True
        except Exception:
            # Provider events can contain conversation text. Do not log payloads
            # or SQL parameter strings when the optional database is unavailable.
            logger.warning("Voice checkpoint unavailable", call_id=record["call_id"])
            return False

    async def load(self, call_id: str) -> dict | None:
        if self.database is None or not self.database.healthy:
            return None
        try:
            async with self.database.session_context() as db:
                row = await db.get(VoiceCallORM, call_id)
                return dict(row.snapshot) if row is not None else None
        except Exception:
            logger.warning("Voice checkpoint lookup unavailable", call_id=call_id)
            return None
