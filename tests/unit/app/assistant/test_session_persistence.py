"""Foreground and background writes retain their order through transaction commit."""

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql.asyncpg import dialect

from assistant_runtime.app.assistant._session_persistence import SessionPersistence
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.services.database.models import SessionORM
from assistant_runtime.services.history.models import MemoryEntry, WorkingMemory


class DelayedCommitDatabase:
    def __init__(self):
        self.first_write = asyncio.Event()
        self.release = asyncio.Event()
        self.entries = 0
        self.stored = {}

    @asynccontextmanager
    async def session_context(self):
        self.entries += 1
        first = self.entries == 1
        values = {}
        stored = self.stored

        class Transaction:
            async def execute(self, statement):
                if statement.is_select:
                    if statement.column_descriptions[0]["entity"] is SessionORM:
                        return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(**stored))
                    return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
                params = copy.deepcopy(statement.compile().params)
                for field in ("working_memory", "pending_action", "usage", "segments"):
                    if field in params:
                        params[field] = json.loads(JSONB().bind_processor(dialect())(params[field]))
                values.update(params)
                return None

            async def flush(self):
                pass

        yield Transaction()
        if first:
            self.first_write.set()
            await self.release.wait()
        self.stored.update(values)


async def test_delayed_background_state_save_cannot_erase_new_pending_action():
    store = SessionStore()
    memory_state = WorkingMemory(
        active_goal="remember", entries=[MemoryEntry(content="an insight", category="other")]
    )
    store.get_context("session")["working_memory"] = memory_state
    db = DelayedCommitDatabase()
    store._db = SessionPersistence(db, 24)

    memory = asyncio.create_task(store.save_session_state_async("session"))
    await db.first_write.wait()
    pending = asyncio.create_task(
        store.set_pending_action(
            "session", tool_call_id="next-call", tool_name="act", assistant_message_id="answer"
        )
    )
    await asyncio.sleep(0)
    db.release.set()
    await asyncio.gather(memory, pending)

    assert db.stored["pending_action"]["tool_call_id"] == "next-call"
    assert db.stored["working_memory"] == memory_state.model_dump(mode="json")
    restored = await store._db.load("session")
    assert restored.working_memory == memory_state
    assert restored.working_memory.entries[0].content == "an insight"
    assert not store._db._write_locks


async def test_working_memory_json_is_validated_before_opening_database_session():
    db = DelayedCommitDatabase()
    db.release.set()
    persistence = SessionPersistence(db, 24)
    await persistence.save_state("session", {"working_memory": {"active_goal": "remember"}})
    loaded = await persistence.load("session")
    assert loaded is not None
    assert loaded.working_memory == WorkingMemory(active_goal="remember")

    transactions = db.entries
    for invalid in ({"goal": "unknown field"}, {"entries": "malformed"}):
        with pytest.raises(ValidationError):
            await persistence.save_state("session", {"working_memory": invalid})
    assert db.entries == transactions
    assert db.stored["working_memory"] == loaded.working_memory.model_dump(mode="json")


async def test_delayed_background_usage_write_cannot_replace_continuation_totals():
    store = SessionStore()
    await store.register_user_message(
        AssistantRequest(id="question", session_id="session", content="q")
    )
    await store.register_assistant_message(
        "session", message_id="answer", parent_id="question", content="a", segments=[], usage=None
    )
    db = DelayedCommitDatabase()
    store._db = SessionPersistence(db, 24)
    auxiliary = {"working_memory": {"input_tokens": 10}}
    background = asyncio.create_task(
        store.update_message("session", "answer", usage={"input_tokens": 1, "auxiliary": auxiliary})
    )
    await db.first_write.wait()
    # A transaction waiting for commit must not publish its tentative values.
    assert store.get_message("session", "answer")["usage"] is None
    continuation = asyncio.create_task(
        store.update_message(
            "session",
            "answer",
            content="continued",
            usage=lambda current: {**current, "input_tokens": 2},
        )
    )
    await asyncio.sleep(0)
    db.release.set()
    await asyncio.gather(background, continuation)

    assert db.stored["usage"] == {"input_tokens": 2, "auxiliary": auxiliary}
    assert db.stored["content"] == "continued"
    assert store.get_message("session", "answer")["usage"] == db.stored["usage"]
    assert not store._db._write_locks
    assert not store._message_write_locks
