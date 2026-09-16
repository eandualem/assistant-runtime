"""Opt-in PostgreSQL schema/commit/reconnect smoke, independent of any model."""

import os
from uuid import uuid4

import pytest
from sqlalchemy import delete

from assistant_runtime.config import AppSettings
from assistant_runtime.services.database.interface import DatabaseService
from assistant_runtime.services.database.models import MessageORM, SessionORM

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DATABASE_SMOKE") != "1", reason="Set RUN_DATABASE_SMOKE=1 for a scratch Postgres"
)


async def test_committed_session_survives_reconnect():
    """A fresh service reads committed session, message and JSONB pending-action data."""
    config = AppSettings().database
    session_id = f"smoke-{uuid4().hex}"
    message_id = f"smoke-{uuid4().hex}"
    pending = {
        "tool_call_id": "smoke-call",
        "tool_name": "smoke_action",
        "assistant_message_id": message_id,
        "batch": ["smoke-call"],
    }
    writer = DatabaseService(config)
    reader = DatabaseService(config)
    committed = False
    try:
        await writer.start()
        assert writer.healthy, "Scratch Postgres must be reachable; degraded mode is not a pass"
        async with writer.session_context() as session:
            session.add(SessionORM(id=session_id, owner_id="smoke-test", pending_action=pending))
            await session.flush()
            session.add(
                MessageORM(
                    id=message_id,
                    session_id=session_id,
                    role="assistant",
                    content="Smoke fixture",
                    segments=[{"type": "text", "content": "Smoke fixture"}],
                    usage={"input_tokens": 1, "output_tokens": 1},
                )
            )
        committed = True
        await writer.stop()

        await reader.start()
        assert reader.healthy
        async with reader.session_context() as session:
            stored = await session.get(SessionORM, session_id)
            message = await session.get(MessageORM, message_id)
            assert stored is not None
            assert stored.owner_id == "smoke-test"
            assert stored.pending_action == pending
            assert message is not None
            assert message.content == "Smoke fixture"
            assert message.usage == {"input_tokens": 1, "output_tokens": 1}
    finally:
        # A committed row must not be silently abandoned if reconnect failed.
        try:
            if committed:
                if not reader.healthy:
                    await reader.stop()
                    await reader.start()
                assert reader.healthy, (
                    f"Cleanup requires a reachable database: session {session_id}"
                )
                async with reader.session_context() as session:
                    await session.execute(delete(SessionORM).where(SessionORM.id == session_id))
        finally:
            await reader.stop()
            await writer.stop()
