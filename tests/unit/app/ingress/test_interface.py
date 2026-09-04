"""Tests for IngressService: delivery into sessions, queueing, draining."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.ingress.interface import IngressService, envelope


def _request(message_id: str, session_id: str = "sess-1", parent_id: str | None = None):
    return AssistantRequest(id=message_id, session_id=session_id, parent_id=parent_id, content="hi")


async def _seeded_store() -> SessionStore:
    store = SessionStore()
    await store.register_user_message(_request("user-1"))
    await store.register_assistant_message(
        "sess-1",
        message_id="assistant-1",
        parent_id="user-1",
        content="ok",
        segments=None,
        usage=None,
    )
    return store


def _service(store: SessionStore | None, *, streaming=None, db=None, sio=None) -> IngressService:
    assistant = MagicMock()
    assistant.get_session_store.return_value = store
    streaming = streaming or MagicMock()
    return IngressService(
        assistant_service=assistant,
        streaming_service=streaming,
        database_service=db,
        socket_server=sio,
    )


class TestEnvelope:
    def test_sender_and_channel(self):
        assert envelope("telegram", "ada", "hi") == "[via:telegram from:ada] hi"

    def test_channel_only_when_sender_is_the_channel(self):
        assert envelope("heartbeat", "heartbeat", "tick") == "[via:heartbeat] tick"


class TestDeliver:
    async def test_promoted_when_the_session_is_idle_and_runs_in_background(self):
        store = await _seeded_store()
        events = [
            {"type": "agent_status", "status": "started"},
            {"type": "agent_status", "status": "completed"},
        ]

        async def _stream(_request):
            for event in events:
                yield event

        streaming = MagicMock()
        streaming.accept_steering = AsyncMock(return_value="promoted")
        streaming.stream_message = _stream
        sio = MagicMock()
        sio.emit = AsyncMock()
        service = _service(store, streaming=streaming, sio=sio)
        await service.start()

        result = await service.deliver(
            from_agent="leo", via="tmux", message="done", session_id="sess-1"
        )
        await asyncio.sleep(0.05)

        assert result == {"status": "delivered", "session_id": "sess-1", "delivery": "promoted"}
        request = streaming.accept_steering.await_args.args[0]
        assert request.is_steering
        assert request.content == "[via:tmux from:leo] done"
        assert sio.emit.await_count == 2
        assert sio.emit.await_args.kwargs["room"] == "session:sess-1"

    async def test_queued_into_a_live_turn_runs_nothing(self):
        store = await _seeded_store()
        streaming = MagicMock()
        streaming.accept_steering = AsyncMock(return_value="queued")
        service = _service(store, streaming=streaming)
        await service.start()

        result = await service.deliver(
            from_agent="leo", via="tmux", message="x", session_id="sess-1"
        )

        assert result["delivery"] == "queued"
        assert not service._background

    async def test_falls_back_to_the_most_recent_session(self):
        store = await _seeded_store()
        streaming = MagicMock()
        streaming.accept_steering = AsyncMock(return_value="queued")
        service = _service(store, streaming=streaming)
        await service.start()

        result = await service.deliver(from_agent="bot", via="inbox", message="note")

        assert result["session_id"] == "sess-1"

    async def test_unknown_session_and_no_sessions_queue_in_memory(self):
        service = _service(SessionStore())
        await service.start()

        result = await service.deliver(
            from_agent="bot", via="inbox", message="note", session_id="nope"
        )

        assert result["status"] == "queued"
        assert result["inbox_id"]
        assert (await service.health_check())["queued"] == 1

    async def test_queued_messages_drain_into_the_next_turn(self):
        service = _service(SessionStore())
        await service.start()
        await service.deliver(from_agent="bot", via="inbox", message="first")
        await service.deliver(
            from_agent="bot", via="inbox", message="for-other", session_id="other"
        )

        store = await _seeded_store()
        service._assistant.get_session_store.return_value = store
        drained = await service.drain("sess-1")

        assert drained == 1
        pending = await store.list_pending_steering("sess-1")
        assert [p["content"] for p in pending] == ["[via:inbox from:bot] first"]
        assert (await service.health_check())["queued"] == 1  # the other session's note waits

    async def test_database_inbox_is_used_when_healthy(self):
        db = MagicMock()
        db.healthy = True
        session = AsyncMock()

        @asynccontextmanager
        async def _ctx():
            yield session

        db.session_context = _ctx
        service = _service(SessionStore(), db=db)
        await service.start()
        row = MagicMock()
        row.id = "inbox-9"
        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.create = AsyncMock(return_value=row)
            mp.setattr(
                "assistant_runtime.services.database.repositories.InboxRepository",
                MagicMock(return_value=repo),
            )
            result = await service.deliver(from_agent="bot", via="inbox", message="note")

        assert result == {"status": "queued", "inbox_id": "inbox-9"}
        assert repo.create.await_args.kwargs["context"] == {"via": "inbox", "injected": True}

    async def test_stop_cancels_background_turns(self):
        store = await _seeded_store()

        async def _stream(_request):
            await asyncio.sleep(10)
            yield {"type": "agent_status", "status": "completed"}

        streaming = MagicMock()
        streaming.accept_steering = AsyncMock(return_value="promoted")
        streaming.stream_message = _stream
        service = _service(store, streaming=streaming)
        await service.start()
        await service.deliver(from_agent="leo", via="tmux", message="x", session_id="sess-1")
        assert len(service._background) == 1

        await service.stop()

        assert not service._background
