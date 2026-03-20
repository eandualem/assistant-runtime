"""Tests for Socket.IO server — AssistantNamespace and create_sio()."""

from __future__ import annotations

import asyncio
import contextlib
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
        ns.enter_room = AsyncMock()
        ns.emit = AsyncMock()

        await ns.on_assistant_join_session("sid-1", {"session_id": "sess-abc"})

        ns.enter_room.assert_awaited_once_with("sid-1", "session:sess-abc")

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
    async def test_cancels_stale_stream_on_new_message(self):
        """When a new message arrives with an active stream, cancel and replace."""
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        # Simulate an active (stuck) stream
        never_done = asyncio.get_event_loop().create_future()
        old_task = asyncio.ensure_future(never_done)
        ns._active_streams["sess-1"] = old_task

        # Mock streaming service for the new message
        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        data = {"session_id": "sess-1", "message": "Second message"}
        await ns.on_assistant_message("sid-1", data)

        # Old task should be cancelled
        assert old_task.cancelled()

        # New task should be created — no error emitted
        new_task = ns._active_streams.get("sess-1")
        assert new_task is not None
        assert new_task is not old_task
        await new_task

        # No conflict error emitted
        error_calls = [
            call for call in ns.emit.call_args_list
            if call.args[0] == "assistant:error" and call.args[1].get("type") == "conflict"
        ]
        assert not error_calls

    @pytest.mark.asyncio
    async def test_continuation_does_not_cancel_active_stream(self):
        """Continuations (tool_call_id set) must not cancel the draining stream."""
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        # Simulate an active stream (draining after deferred tool dispatch)
        never_done = asyncio.get_event_loop().create_future()
        old_task = asyncio.ensure_future(never_done)
        ns._active_streams["sess-1"] = old_task

        # Mock streaming service
        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "final_response", "content": "Done", "model": "m"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        # Send a continuation (has tool_call_id)
        data = {
            "session_id": "sess-1",
            "message": "",
            "tool_call_id": "call_123",
            "tool_result": {"success": True},
        }
        await ns.on_assistant_message("sid-1", data)

        # Old task should NOT be cancelled — continuations don't cancel
        assert not old_task.cancelled()

        # New task still created (alongside the old draining one)
        new_task = ns._active_streams.get("sess-1")
        assert new_task is not None
        await new_task

        # Cleanup
        never_done.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await old_task

    @pytest.mark.asyncio
    async def test_discards_completed_task_before_duplicate_check(self):
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        done_task = asyncio.get_event_loop().create_future()
        done_task.set_result(None)
        ns._active_streams["sess-1"] = asyncio.ensure_future(done_task)

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        await ns.on_assistant_message("sid-1", {"session_id": "sess-1", "message": "Retry"})

        assert "sess-1" in ns._active_streams
        emitted = [call.args[0] for call in ns.emit.call_args_list]
        assert "assistant:error" not in emitted

        task = ns._active_streams["sess-1"]
        await task
        await asyncio.sleep(0)
        assert "sess-1" not in ns._active_streams

    @pytest.mark.asyncio
    async def test_second_message_after_tool_call_stream(self):
        """Reproduce #16: backend tool call stream completes, then second message is sent.

        After a stream with look_at_screen (backend tool) completes normally,
        the session must be released so the next message can start.
        """
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        call_count = 0

        mock_service = MagicMock()

        async def mock_stream(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First message: backend tool call (look_at_screen)
                yield {"type": "agent_status", "status": "started"}
                yield {"type": "tool_call", "tool_name": "look_at_screen", "arguments": {}, "call_id": "c1"}
                yield {"type": "tool_result", "tool_name": "look_at_screen", "result": "[image]", "call_id": "c1"}
                yield {"type": "text_delta", "content": "I can see the agents page."}
                yield {"type": "final_response", "content": "I can see the agents page.", "model": "m"}
                yield {"type": "agent_status", "status": "completed"}
            else:
                # Second message: simple response
                yield {"type": "agent_status", "status": "started"}
                yield {"type": "text_delta", "content": "Sure!"}
                yield {"type": "final_response", "content": "Sure!", "model": "m"}
                yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        # First message
        await ns.on_assistant_message("sid-1", {"session_id": "sess-1", "message": "Look at my screen"})
        task1 = ns._active_streams.get("sess-1")
        assert task1 is not None
        await task1
        await asyncio.sleep(0)  # let done_callback fire

        # Session must be released
        assert "sess-1" not in ns._active_streams, (
            "Session still in _active_streams after first stream completed"
        )

        # Second message — must NOT get "Stream already active"
        ns.emit.reset_mock()
        await ns.on_assistant_message("sid-1", {"session_id": "sess-1", "message": "Navigate to workspace"})
        task2 = ns._active_streams.get("sess-1")
        assert task2 is not None, "Second message should have created a new task"
        await task2
        await asyncio.sleep(0)

        # Verify no error was emitted
        error_calls = [
            call for call in ns.emit.call_args_list
            if call.args[0] == "assistant:error"
        ]
        assert not error_calls, f"Unexpected error: {error_calls}"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_second_message_after_deferred_tool_stream(self):
        """Reproduce #16: deferred tool call, then continuation, then new message.

        Full cycle: message → deferred final_response → continuation → completed → new message.
        """
        ns = AssistantNamespace("/assistant")
        ns.emit = AsyncMock()

        call_count = 0

        mock_service = MagicMock()

        async def mock_stream(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First message → deferred tool call
                yield {"type": "agent_status", "status": "started"}
                yield {"type": "tool_call", "tool_name": "look_at_screen", "arguments": {}, "call_id": "c1"}
                yield {
                    "type": "final_response",
                    "content": None,
                    "model": "m",
                    "pending_tool_call": {"call_id": "c1", "tool_name": "look_at_screen", "arguments": {}},
                }
                yield {"type": "agent_status", "status": "completed"}
            elif call_count == 2:
                # Continuation
                yield {"type": "agent_status", "status": "started"}
                yield {"type": "text_delta", "content": "I see the page."}
                yield {"type": "final_response", "content": "I see the page.", "model": "m"}
                yield {"type": "agent_status", "status": "completed"}
            else:
                # Third message
                yield {"type": "agent_status", "status": "started"}
                yield {"type": "final_response", "content": "Done.", "model": "m"}
                yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server

        # Step 1: First message → deferred tool call
        await ns.on_assistant_message("sid-1", {"session_id": "sess-1", "message": "Look at screen"})
        task1 = ns._active_streams.get("sess-1")
        assert task1 is not None
        await task1
        await asyncio.sleep(0)

        # Session should be released (early-release at deferred final_response)
        assert "sess-1" not in ns._active_streams, (
            f"Session still active after deferred stream. "
            f"_active_streams keys: {list(ns._active_streams.keys())}"
        )

        # Step 2: Continuation (tool result from frontend)
        ns.emit.reset_mock()
        await ns.on_assistant_message("sid-1", {
            "session_id": "sess-1",
            "message": "",
            "tool_call_id": "c1",
            "tool_result": {"screenshot": "data:image/jpeg;base64,abc"},
        })
        task2 = ns._active_streams.get("sess-1")
        assert task2 is not None, "Continuation should have created a task"
        await task2
        await asyncio.sleep(0)

        assert "sess-1" not in ns._active_streams

        # Step 3: New message after full cycle
        ns.emit.reset_mock()
        await ns.on_assistant_message("sid-1", {"session_id": "sess-1", "message": "Navigate to workspace"})
        task3 = ns._active_streams.get("sess-1")
        assert task3 is not None, "Third message should have created a task"
        await task3
        await asyncio.sleep(0)

        error_calls = [
            call for call in ns.emit.call_args_list
            if call.args[0] == "assistant:error"
        ]
        assert not error_calls, f"Unexpected error on third message: {error_calls}"
        assert call_count == 3

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

    @pytest.mark.asyncio
    async def test_releases_session_before_completed_emit(self):
        ns = AssistantNamespace("/assistant")
        emitted: list[tuple[str, dict]] = []

        async def emit(event_name, payload, **kwargs):
            if event_name == "assistant:status" and payload.get("status") == "completed":
                assert "sess-1" not in ns._active_streams
            emitted.append((event_name, payload))

        ns.emit = AsyncMock(side_effect=emit)

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "final_response", "content": None, "model": "m"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server
        ns._active_streams["sess-1"] = asyncio.current_task()

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        assert [name for name, _ in emitted] == [
            "assistant:status",
            "assistant:final_response",
            "assistant:status",
        ]
        assert "sess-1" not in ns._active_streams

    @pytest.mark.asyncio
    async def test_releases_session_before_deferred_final_response_emit(self):
        """Session must be released before emitting final_response with pending_tool_call.

        The client fires the continuation immediately upon receiving final_response
        with pending_tool_call. If the session is still in _active_streams at that
        point, the continuation is rejected with 'Stream already active'.
        """
        ns = AssistantNamespace("/assistant")
        emitted: list[tuple[str, dict]] = []

        async def emit(event_name, payload, **kwargs):
            if (
                event_name == "assistant:final_response"
                and payload.get("pending_tool_call") is not None
            ):
                # Session must already be released when this event is emitted
                assert "sess-1" not in ns._active_streams
            emitted.append((event_name, payload))

        ns.emit = AsyncMock(side_effect=emit)

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {
                "type": "final_response",
                "content": None,
                "model": "m",
                "pending_tool_call": {
                    "call_id": "call_123",
                    "tool_name": "look_at_screen",
                    "arguments": {},
                },
            }
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server
        ns._active_streams["sess-1"] = asyncio.current_task()

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        assert [name for name, _ in emitted] == [
            "assistant:status",
            "assistant:final_response",
            "assistant:status",
        ]
        assert "sess-1" not in ns._active_streams

    @pytest.mark.asyncio
    async def test_no_early_release_for_normal_final_response(self):
        """Normal final_response (no pending_tool_call) does NOT early-release."""
        ns = AssistantNamespace("/assistant")
        emitted: list[tuple[str, dict]] = []

        async def emit(event_name, payload, **kwargs):
            if event_name == "assistant:final_response" and "pending_tool_call" not in payload:
                # Session should still be active — only released at completed
                assert "sess-1" in ns._active_streams
            emitted.append((event_name, payload))

        ns.emit = AsyncMock(side_effect=emit)

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "final_response", "content": "Hello!", "model": "m"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server
        ns._active_streams["sess-1"] = asyncio.current_task()

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        assert "sess-1" not in ns._active_streams

    @pytest.mark.asyncio
    async def test_completed_does_not_pop_continuation_task(self):
        """After deferred early-release, completed must not pop a newer continuation task.

        Race condition: deferred final_response releases session → client sends
        continuation → new task registered → old generator yields completed →
        completed must NOT pop the new task.
        """
        ns = AssistantNamespace("/assistant")
        continuation_task = asyncio.get_event_loop().create_future()
        continuation_task_obj = asyncio.ensure_future(continuation_task)

        async def emit(event_name, payload, **kwargs):
            # Simulate: after deferred final_response is emitted, a continuation
            # task registers itself before completed arrives.
            if (
                event_name == "assistant:final_response"
                and payload.get("pending_tool_call") is not None
            ):
                # Continuation starts and registers its task
                ns._active_streams["sess-1"] = continuation_task_obj

        ns.emit = AsyncMock(side_effect=emit)

        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {
                "type": "final_response",
                "content": None,
                "model": "m",
                "pending_tool_call": {"call_id": "c1", "tool_name": "ui_send_event", "arguments": {}},
            }
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        mock_server = MagicMock()
        mock_server.fastapi_app.state.streaming_service = mock_service
        ns.server = mock_server
        ns._active_streams["sess-1"] = asyncio.current_task()

        request = AssistantRequest(session_id="sess-1", message="Hello")
        await ns._run_stream("sid-1", "sess-1", request)

        # The continuation's task must still be in _active_streams
        assert ns._active_streams.get("sess-1") is continuation_task_obj

        # Cleanup
        continuation_task.set_result(None)
        await continuation_task_obj

    @pytest.mark.asyncio
    async def test_done_callback_does_not_pop_continuation_task(self):
        """done_callback from old task must not remove a newer continuation task."""
        ns = AssistantNamespace("/assistant")

        # Simulate: old task finishes, but continuation already registered
        old_future = asyncio.get_event_loop().create_future()
        old_task = asyncio.ensure_future(old_future)
        ns._active_streams["sess-1"] = old_task

        # Register the done_callback (same as on_assistant_message does)
        def _done_cleanup(_t):
            if ns._active_streams.get("sess-1") is _t:
                ns._active_streams.pop("sess-1", None)

        old_task.add_done_callback(_done_cleanup)

        # Continuation starts and replaces the entry
        new_future = asyncio.get_event_loop().create_future()
        new_task = asyncio.ensure_future(new_future)
        ns._active_streams["sess-1"] = new_task

        # Old task completes — callback should NOT remove the new task
        old_future.set_result(None)
        await asyncio.sleep(0)  # let callback fire

        assert ns._active_streams.get("sess-1") is new_task

        # Cleanup
        new_future.set_result(None)
        await new_task
