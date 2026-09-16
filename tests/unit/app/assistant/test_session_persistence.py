"""Foreground and background writes retain their order through transaction commit."""

import asyncio
import copy
from contextlib import asynccontextmanager

from assistant_runtime.app.assistant._session_persistence import SessionPersistence
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.models import AssistantRequest


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

        class Transaction:
            async def execute(self, statement):
                values.update(copy.deepcopy(statement.compile().params))

            async def flush(self):
                pass

        yield Transaction()
        if first:
            self.first_write.set()
            await self.release.wait()
        self.stored.update(values)


async def test_delayed_background_state_save_cannot_erase_new_pending_action():
    store = SessionStore()
    store.get_context("session")["working_memory"] = {"goal": "remember"}
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
    assert db.stored["working_memory"] == {"goal": "remember"}
    assert not store._db._write_locks


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
    continuation = asyncio.create_task(
        store.update_message(
            "session",
            "answer",
            content="continued",
            usage={"input_tokens": 2, "auxiliary": auxiliary},
        )
    )
    await asyncio.sleep(0)
    db.release.set()
    await asyncio.gather(background, continuation)

    assert db.stored["usage"] == {"input_tokens": 2, "auxiliary": auxiliary}
    assert db.stored["content"] == "continued"
    assert store.get_message("session", "answer")["usage"] == db.stored["usage"]
    assert not store._db._write_locks
