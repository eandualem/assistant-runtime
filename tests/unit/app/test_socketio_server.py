from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import socketio

from lovely_assistant.app.socketio_server import _EVENT_TYPE_MAP, AssistantNamespace, create_sio


def _server_with_streaming_service(streaming_service: MagicMock) -> MagicMock:
    server = MagicMock()
    server.fastapi_app.state.streaming_service = streaming_service
    return server


async def _never_finishes() -> None:
    await asyncio.Event().wait()


class TestCreateSio:
    def test_returns_async_server_with_assistant_namespace(self) -> None:
        sio = create_sio()

        assert isinstance(sio, socketio.AsyncServer)
        assert isinstance(sio.namespace_handlers["/assistant"], AssistantNamespace)


class TestEventMap:
    def test_all_events_use_assistant_prefix(self) -> None:
        assert all(value.startswith("assistant:") for value in _EVENT_TYPE_MAP.values())


class TestAssistantNamespaceJoin:
    @pytest.mark.asyncio
    async def test_join_session_requires_session_id(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()

        await namespace.on_assistant_join_session("sid-1", {})

        namespace.emit.assert_awaited_once_with(
            "assistant:error",
            {"type": "validation", "message": "Missing session_id"},
            to="sid-1",
        )


class TestAssistantNamespaceMessages:
    @pytest.mark.asyncio
    async def test_invalid_payload_emits_validation_error(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()

        await namespace.on_assistant_message("sid-1", "bad-payload")

        namespace.emit.assert_awaited_once_with(
            "assistant:error",
            {"type": "validation", "message": "Invalid assistant request payload"},
            to="sid-1",
        )

    @pytest.mark.asyncio
    async def test_starts_stream_for_unified_message_contract(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()

        async def _stream(_request):
            yield {"type": "agent_status", "status": "started"}
            yield {
                "type": "final_response",
                "content": "Done",
                "model": "openai:gpt-5.4",
                "message_id": "assistant-1",
            }
            yield {"type": "agent_status", "status": "completed"}

        streaming_service = MagicMock()
        streaming_service.stream_message = _stream
        streaming_service.queue_guidance = AsyncMock()
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "user-1",
                "session_id": "sess-1",
                "parent_id": None,
                "content": "Hi",
            },
        )

        task = namespace._active_streams["sess-1"]
        await task
        assert namespace.emit.await_count == 3

    @pytest.mark.asyncio
    async def test_new_message_cancels_stale_stream(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()

        old_task = asyncio.create_task(_never_finishes())
        namespace._active_streams["sess-1"] = old_task

        async def _stream(_request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "agent_status", "status": "completed"}

        streaming_service = MagicMock()
        streaming_service.stream_message = _stream
        streaming_service.queue_guidance = AsyncMock()
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "user-2",
                "session_id": "sess-1",
                "parent_id": "assistant-1",
                "content": "Retry",
            },
        )

        assert old_task.cancelling() > 0
        await namespace._active_streams["sess-1"]

    @pytest.mark.asyncio
    async def test_continuation_does_not_cancel_active_stream(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()

        old_task = asyncio.create_task(_never_finishes())
        namespace._active_streams["sess-1"] = old_task

        async def _stream(_request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "agent_status", "status": "completed"}

        streaming_service = MagicMock()
        streaming_service.stream_message = _stream
        streaming_service.queue_guidance = AsyncMock()
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "continuation-1",
                "session_id": "sess-1",
                "parent_id": "assistant-1",
                "content": "",
                "tool_call_id": "call-1",
                "tool_result": {"ok": True},
            },
        )

        assert old_task.cancelled() is False
        new_task = namespace._active_streams["sess-1"]
        assert new_task is not old_task
        await new_task
        old_task.cancel()

    @pytest.mark.asyncio
    async def test_guidance_is_queued_without_starting_stream(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()
        streaming_service = MagicMock()
        streaming_service.queue_guidance = AsyncMock()
        streaming_service.stream_message = AsyncMock()
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "guidance-1",
                "session_id": "sess-1",
                "parent_id": "assistant-1",
                "content": "Focus on Leo",
                "message_type": "guidance",
            },
        )

        streaming_service.queue_guidance.assert_awaited_once()
        streaming_service.stream_message.assert_not_called()
        assert namespace._active_streams == {}
