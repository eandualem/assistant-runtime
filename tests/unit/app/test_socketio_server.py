"""Tests for Socket.IO server — AssistantNamespace and create_sio()."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import socketio

from lovely_assistant.app.assistant.models import AssistantRequest
from lovely_assistant.app.socketio_server import (
    _EVENT_TYPE_MAP,
    AssistantNamespace,
    create_sio,
)


class TestCreateSio:
    def test_returns_async_server(self):
        sio = create_sio()
        assert isinstance(sio, socketio.AsyncServer)

    def test_namespace_registered(self):
        sio = create_sio()
        # The namespace handler should be registered at /assistant
        handler = sio.namespace_handlers.get("/assistant")
        assert handler is not None
        assert isinstance(handler, AssistantNamespace)


class TestEventTypeMap:
    def test_all_protocol_events_mapped(self):
        expected = {
            "agent_status",
            "thinking_delta",
            "text_delta",
            "tool_call",
            "tool_result",
            "tool_error",
            "final_response",
            "error",
        }
        assert set(_EVENT_TYPE_MAP.keys()) == expected

    def test_debug_not_in_map(self):
        # debug_* events are handled dynamically, not in the static map
        for key in _EVENT_TYPE_MAP:
            assert not key.startswith("debug_")

    def test_all_values_prefixed(self):
        for value in _EVENT_TYPE_MAP.values():
            assert value.startswith("assistant:")


class TestOnConnect:
    @pytest.mark.asyncio
    async def test_returns_true(self):
        ns = AssistantNamespace("/assistant")
        result = await ns.on_connect("sid-1", {})
        assert result is True

    @pytest.mark.asyncio
    async def test_with_auth(self):
        ns = AssistantNamespace("/assistant")
        result = await ns.on_connect("sid-1", {}, auth={"token": "abc"})
        assert result is True


class TestOnDisconnect:
    @pytest.mark.asyncio
    async def test_logs_without_error(self):
        ns = AssistantNamespace("/assistant")
        # Should not raise
        await ns.on_disconnect("sid-1")


class TestOnJoinSession:
    @pytest.mark.asyncio
    async def test_enters_room(self):
        ns = AssistantNamespace("/assistant")
        ns.enter_room = MagicMock()
        ns.emit = AsyncMock()

        await ns.on_assistant_join_session("sid-1", {"session_id": "sess-abc"})

        ns.enter_room.assert_called_once_with("sid-1", "session:sess-abc")

    @pytest.mark.asyncio
    async def test_error_on_missing_session_id(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        await ns.on_assistant_join_session("sid-1", {})

        ns.emit.assert_called_once_with(
            "assistant:error",
            {"type": "validation", "message": "Missing session_id"},
            to="sid-1",
        )

    @pytest.mark.asyncio
    async def test_error_on_non_dict_data(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        await ns.on_assistant_join_session("sid-1", "not-a-dict")

        ns.emit.assert_called_once()
        args = ns.emit.call_args
        assert args[0][0] == "assistant:error"


class TestOnAssistantMessage:
    @pytest.mark.asyncio
    async def test_starts_task(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        # Mock streaming service
        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "text_delta", "content": "Hello"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream

        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        data = {"session_id": "sess-1", "message": "Hi"}
        await ns.on_assistant_message("sid-1", data)

        # Task should be created
        assert "sess-1" in ns._active_streams

        # Wait for task to complete
        await ns._active_streams["sess-1"]

        # Events were emitted via to=sid
        assert ns.emit.call_count >= 3

    @pytest.mark.asyncio
    async def test_rejects_duplicate_stream(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        # Simulate an active stream
        never_done = asyncio.get_event_loop().create_future()
        ns._active_streams["sess-1"] = asyncio.ensure_future(never_done)

        data = {"session_id": "sess-1", "message": "Second message"}
        await ns.on_assistant_message("sid-1", data)

        ns.emit.assert_called_once_with(
            "assistant:error",
            {"type": "conflict", "message": "Stream already active for this session"},
            to="sid-1",
        )

        # Cleanup
        ns._active_streams["sess-1"].cancel()
        with pytest.raises((asyncio.CancelledError, Exception)):
            await ns._active_streams["sess-1"]

    @pytest.mark.asyncio
    async def test_validates_request_missing_fields(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        await ns.on_assistant_message("sid-1", {"bad": "data"})

        ns.emit.assert_called_once()
        args = ns.emit.call_args
        assert args[0][0] == "assistant:error"
        assert args[0][1]["type"] == "validation"

    @pytest.mark.asyncio
    async def test_catches_pydantic_validation_error(self):
        """Pydantic validation errors (e.g., invalid config) are caught and reported."""
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        # config with extra="forbid" should reject unknown fields
        data = {
            "session_id": "sess-1",
            "message": "Hello",
            "config": {"not_a_real_field": "boom"},
        }
        await ns.on_assistant_message("sid-1", data)

        ns.emit.assert_called_once()
        args = ns.emit.call_args
        assert args[0][0] == "assistant:error"
        assert args[0][1]["type"] == "validation"
        assert args[1]["to"] == "sid-1"


class TestOnAssistantCancel:
    @pytest.mark.asyncio
    async def test_cancels_task(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        async def long_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(long_task())
        ns._active_streams["sess-1"] = task

        await ns.on_assistant_cancel("sid-1", {"session_id": "sess-1"})

        # Give the event loop a cycle to process cancellation
        await asyncio.sleep(0)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_noop_on_missing_stream(self):
        ns = AssistantNamespace("/assistant")
        # Should not raise
        await ns.on_assistant_cancel("sid-1", {"session_id": "no-such-session"})

    @pytest.mark.asyncio
    async def test_noop_on_missing_session_id(self):
        ns = AssistantNamespace("/assistant")
        await ns.on_assistant_cancel("sid-1", {})


class TestRunStream:
    @pytest.mark.asyncio
    async def test_emits_events_to_sid(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "text_delta", "content": "Hi"}
            yield {"type": "final_response", "content": "Hi", "model": "m"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        # Verify events were emitted to sid
        calls = ns.emit.call_args_list
        assert len(calls) == 4

        # Check event name mapping
        assert calls[0][0][0] == "assistant:status"
        assert calls[1][0][0] == "assistant:text_delta"
        assert calls[2][0][0] == "assistant:final_response"
        assert calls[3][0][0] == "assistant:status"

        # Check sid targeting (to=sid, not room=room)
        for call in calls:
            assert call[1]["to"] == "sid-1"

    @pytest.mark.asyncio
    async def test_maps_debug_events(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "debug_request", "data": "something"}
            yield {"type": "debug_usage", "tokens": 100}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        calls = ns.emit.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "assistant:debug"
        assert calls[1][0][0] == "assistant:debug"

    @pytest.mark.asyncio
    async def test_handles_cancellation(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            raise asyncio.CancelledError()

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        # Should have emitted the start event + cancellation error
        calls = ns.emit.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == "assistant:status"
        assert calls[1][0][0] == "assistant:error"
        assert calls[1][0][1]["type"] == "cancelled"
        # Error emitted to sid
        assert calls[1][1]["to"] == "sid-1"

    @pytest.mark.asyncio
    async def test_handles_exception(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        mock_service = MagicMock()

        async def mock_stream(request):
            raise RuntimeError("LLM exploded")
            yield  # noqa: RET504 — make it an async generator

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        calls = ns.emit.call_args_list
        assert len(calls) == 1
        assert calls[0][0][0] == "assistant:error"
        assert calls[0][0][1]["type"] == "internal"
        assert "LLM exploded" in calls[0][0][1]["message"]
        # Error emitted to sid
        assert calls[0][1]["to"] == "sid-1"

    @pytest.mark.asyncio
    async def test_cleans_up_task_reference(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        data = {"session_id": "sess-cleanup", "message": "Hi"}
        await ns.on_assistant_message("sid-1", data)

        # Wait for task
        task = ns._active_streams.get("sess-cleanup")
        if task:
            await task

        # After completion, done_callback should have removed it
        # Give event loop a cycle for callback
        await asyncio.sleep(0)
        assert "sess-cleanup" not in ns._active_streams
