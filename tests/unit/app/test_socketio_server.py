from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import socketio

from assistant_runtime.app.socketio_server import _EVENT_TYPE_MAP, AssistantNamespace, create_sio
from assistant_runtime.app.streaming.interface import StreamingService


def _server_with_streaming_service(streaming_service: MagicMock) -> MagicMock:
    server = MagicMock()
    server.fastapi_app.state.streaming_service = streaming_service
    return server


async def _never_finishes() -> None:
    await asyncio.Event().wait()


def _namespace_for_stream(stream):
    service = MagicMock()
    service.stream_message = stream
    service.cancel_session = AsyncMock(return_value=True)
    service.wait_for_session = AsyncMock()
    service.cancelled_before_start_events = StreamingService.cancelled_before_start_events
    namespace = AssistantNamespace("/assistant")
    namespace.server = _server_with_streaming_service(service)
    namespace.emit = AsyncMock()
    return namespace, service


def _message(message_id):
    return {"id": message_id, "session_id": "sess-1", "content": "Help"}


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

    @pytest.mark.asyncio
    async def test_join_session_normalises_camel_case_host_context(self) -> None:
        streaming = MagicMock()
        streaming.warm_session = AsyncMock()
        namespace = AssistantNamespace("/assistant")
        namespace.server = _server_with_streaming_service(streaming)
        namespace.emit = AsyncMock()
        namespace.enter_room = AsyncMock()

        await namespace.on_assistant_join_session(
            "sid-1",
            {
                "session_id": "sess-1",
                "hostContext": {"page": {"name": "tasks", "data": {"activeFilters": []}}},
            },
        )

        streaming.warm_session.assert_awaited_once()
        session_id, context = streaming.warm_session.await_args.args
        assert session_id == "sess-1"
        assert context["version"] == 1
        assert context["view"] == {
            "name": "tasks",
            "description": "",
            "data": {"active_filters": []},
            "state": {},
        }

    @pytest.mark.asyncio
    async def test_join_session_rejects_invalid_host_context(self) -> None:
        streaming = MagicMock()
        streaming.warm_session = AsyncMock()
        namespace = AssistantNamespace("/assistant")
        namespace.server = _server_with_streaming_service(streaming)
        namespace.emit = AsyncMock()
        namespace.enter_room = AsyncMock()

        await namespace.on_assistant_join_session(
            "sid-1", {"session_id": "sess-1", "host_context": {"surprise": 1}}
        )

        streaming.warm_session.assert_not_awaited()
        namespace.enter_room.assert_not_awaited()
        event, payload = namespace.emit.await_args.args
        assert event == "assistant:error"
        assert payload["type"] == "validation"
        assert "surprise" in payload["message"]

    @pytest.mark.asyncio
    async def test_join_session_without_context_warms_with_none(self) -> None:
        streaming = MagicMock()
        streaming.warm_session = AsyncMock()
        namespace = AssistantNamespace("/assistant")
        namespace.server = _server_with_streaming_service(streaming)
        namespace.emit = AsyncMock()
        namespace.enter_room = AsyncMock()

        await namespace.on_assistant_join_session("sid-1", {"session_id": "sess-1"})

        streaming.warm_session.assert_awaited_once_with("sess-1", None)


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
        streaming_service.accept_steering = AsyncMock()
        streaming_service.cancel_session = AsyncMock(return_value=True)
        streaming_service.wait_for_session = AsyncMock()
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
        streaming_service.accept_steering = AsyncMock()
        streaming_service.cancel_session = AsyncMock(return_value=True)
        streaming_service.wait_for_session = AsyncMock()
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
        assert old_task.done()
        streaming_service.cancel_session.assert_awaited_once_with("sess-1")
        streaming_service.wait_for_session.assert_awaited_once_with("sess-1")
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
        streaming_service.accept_steering = AsyncMock()
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
        await asyncio.gather(old_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_steering_is_queued_without_starting_stream(self) -> None:
        namespace = AssistantNamespace("/assistant")
        namespace.emit = AsyncMock()
        streaming_service = MagicMock()
        streaming_service.accept_steering = AsyncMock(return_value="queued")
        streaming_service.stream_message = AsyncMock()
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "steering-1",
                "session_id": "sess-1",
                "content": "Focus on Leo",
                "message_type": "steering",
            },
        )

        streaming_service.accept_steering.assert_awaited_once()
        streaming_service.stream_message.assert_not_called()
        assert namespace._active_streams == {}

    @pytest.mark.asyncio
    async def test_promoted_steering_starts_stream_immediately(self) -> None:
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
        streaming_service.accept_steering = AsyncMock(return_value="promoted")
        streaming_service.stream_message = _stream
        namespace.server = _server_with_streaming_service(streaming_service)

        await namespace.on_assistant_message(
            "sid-1",
            {
                "id": "steering-1",
                "session_id": "sess-1",
                "content": "Focus on Leo",
                "message_type": "steering",
            },
        )

        task = namespace._active_streams["sess-1"]
        await task
        streaming_service.accept_steering.assert_awaited_once()
        assert namespace.emit.await_count == 3


class TestAssistantNamespaceCancellation:
    @pytest.mark.parametrize("waiting", [False, True], ids=["unscheduled", "waiting-predecessor"])
    async def test_cancel_before_first_event_cannot_leave_a_scheduled_turn(self, waiting):
        entered, predecessor_finished, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        produced = []

        async def stream(_request):
            try:
                entered.set()
                await predecessor_finished.wait()
                produced.append("new turn")
                yield {"type": "agent_status", "status": "started"}
            finally:
                closed.set()

        namespace, service = _namespace_for_stream(stream)
        service.cancel_session.return_value = waiting
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]
        try:
            async with asyncio.timeout(5):
                if waiting:
                    await entered.wait()
                await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})
                await namespace._active_streams["sess-1"]
                predecessor_finished.set()
                assert task.done()
                assert produced == []
                assert closed.is_set() is waiting
                assert namespace._active_streams == {}
                assert namespace._stream_delivery == {}
                service.cancel_session.assert_awaited_once_with("sess-1")
                service.wait_for_session.assert_awaited_once_with("sess-1")
                assert [call.args[1] for call in namespace.emit.await_args_list] == (
                    StreamingService.cancelled_before_start_events("sess-1")
                )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_explicit_cancel_reaches_runtime_while_transport_is_emitting(self):
        emitting, release_emit, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def stream(_request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "text_delta", "content": "Partial"}
            await cancelled.wait()
            yield {"type": "agent_status", "status": "completed"}

        namespace, service = _namespace_for_stream(stream)

        async def emit(name, event, **kwargs):
            if name == "assistant:text_delta":
                emitting.set()
                await release_emit.wait()

        async def cancel(session_id):
            cancelled.set()
            return True

        namespace.emit = AsyncMock(side_effect=emit)
        service.cancel_session.side_effect = cancel
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]
        try:
            async with asyncio.timeout(5):
                await emitting.wait()
                await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})
                service.cancel_session.assert_awaited_once_with("sess-1")
                assert task.cancelling() == 0
                assert not task.done()
                release_emit.set()
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_cancel_still_closes_lifecycle_when_producer_finishes_without_events(self):
        entered, finish = asyncio.Event(), asyncio.Event()

        async def stream(_request):
            entered.set()
            await finish.wait()
            return
            yield  # pragma: no cover

        namespace, service = _namespace_for_stream(stream)
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]

        async def cancel(session_id):
            finish.set()
            await task
            return True

        service.cancel_session.side_effect = cancel
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                await namespace.on_assistant_cancel("sid-2", {"session_id": "sess-1"})
                await namespace._active_streams["sess-1"]
                assert task.done()
                assert namespace._stream_delivery == {}
                assert [call.args[1] for call in namespace.emit.await_args_list] == (
                    StreamingService.cancelled_before_start_events("sess-1")
                )
                assert all(call.kwargs["to"] == "sid-1" for call in namespace.emit.await_args_list)
                service.wait_for_session.assert_awaited_once_with("sess-1")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.parametrize("already_emitting", [False, True], ids=["race", "blocked"])
    async def test_cancel_does_not_duplicate_a_started_event_in_flight(self, already_emitting):
        start, emitting, release_emit = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cancelled = asyncio.Event()

        async def stream(_request):
            await start.wait()
            yield {"type": "agent_status", "status": "started"}
            await cancelled.wait()
            for event in StreamingService.cancelled_before_start_events("sess-1")[1:]:
                yield event

        namespace, service = _namespace_for_stream(stream)

        async def emit(name, event, **kwargs):
            if event.get("status") == "started":
                emitting.set()
                await release_emit.wait()

        async def cancel(session_id):
            start.set()
            await emitting.wait()
            cancelled.set()
            return True

        namespace.emit = AsyncMock(side_effect=emit)
        service.cancel_session.side_effect = cancel
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]
        try:
            async with asyncio.timeout(5):
                if already_emitting:
                    start.set()
                    await emitting.wait()
                await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})
                assert task.cancelling() == 0
                assert not task.done()
                release_emit.set()
                await task
                assert [call.args[1] for call in namespace.emit.await_args_list] == (
                    StreamingService.cancelled_before_start_events("sess-1")
                )
                service.wait_for_session.assert_not_awaited()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_debug_event_does_not_count_as_started_for_cancellation(self):
        emitted = asyncio.Event()

        async def stream(_request):
            yield {"type": "debug_request", "message": "Preparing"}
            emitted.set()
            await asyncio.Event().wait()

        namespace, _service = _namespace_for_stream(stream)
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]
        try:
            async with asyncio.timeout(5):
                await emitted.wait()
                await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})
                await namespace._active_streams["sess-1"]
                assert task.done()
                assert [call.args[1] for call in namespace.emit.await_args_list[1:]] == (
                    StreamingService.cancelled_before_start_events("sess-1")
                )
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_blocked_early_cancellation_envelope_does_not_block_replacement(self):
        emitting, replaced = asyncio.Event(), asyncio.Event()

        async def stream(request):
            assert request.id == "user-2"
            replaced.set()
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "agent_status", "status": "completed"}

        namespace, service = _namespace_for_stream(stream)

        async def emit(name, event, **kwargs):
            if event.get("status") == "started" and not emitting.is_set():
                emitting.set()
                await asyncio.Event().wait()

        namespace.emit = AsyncMock(side_effect=emit)
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        original = namespace._active_streams["sess-1"]
        fallback = None
        try:
            async with asyncio.timeout(5):
                await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})
                fallback = namespace._active_streams["sess-1"]
                assert fallback is not original
                await emitting.wait()
                assert not fallback.done()
                await namespace.on_assistant_message("sid-1", _message("user-2"))
                replacement = namespace._active_streams["sess-1"]
                await replacement
                assert replaced.is_set()
                assert original.done()
                assert fallback.done()
                assert service.cancel_session.await_count == 2
                assert service.wait_for_session.await_count == 2
        finally:
            tasks = [original, *namespace._active_streams.values()]
            if fallback is not None:
                tasks.append(fallback)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_explicit_cancel_also_finds_runs_released_by_the_transport(self):
        namespace, service = _namespace_for_stream(None)

        await namespace.on_assistant_cancel("sid-1", {"session_id": "sess-1"})

        service.cancel_session.assert_awaited_once_with("sess-1")

    async def test_replacement_waits_for_native_drain_before_closing_old_transport(self):
        emitting, drain_started, drained = asyncio.Event(), asyncio.Event(), asyncio.Event()
        old_closed, new_started = asyncio.Event(), asyncio.Event()

        async def stream(request):
            if request.id == "user-1":
                try:
                    yield {"type": "text_delta", "content": "Partial"}
                    await asyncio.Event().wait()
                finally:
                    old_closed.set()
            else:
                assert drained.is_set()
                assert old_closed.is_set()
                new_started.set()
                yield {"type": "agent_status", "status": "completed"}

        namespace, service = _namespace_for_stream(stream)

        async def emit(name, event, **kwargs):
            if name == "assistant:text_delta":
                emitting.set()
                await asyncio.Event().wait()

        async def wait_for_session(session_id):
            drain_started.set()
            await drained.wait()

        namespace.emit = AsyncMock(side_effect=emit)
        service.wait_for_session.side_effect = wait_for_session
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        old_task = namespace._active_streams["sess-1"]
        replacement = None
        try:
            async with asyncio.timeout(5):
                await emitting.wait()
                replacement = asyncio.create_task(
                    namespace.on_assistant_message("sid-1", _message("user-2"))
                )
                await drain_started.wait()
                assert not old_closed.is_set()
                assert not new_started.is_set()
                assert old_task.cancelling() == 0
                drained.set()
                await replacement
                await new_started.wait()
                assert old_task.done()
                service.cancel_session.assert_awaited_once_with("sess-1")
        finally:
            tasks = [
                task
                for task in (old_task, replacement, *namespace._active_streams.values())
                if task is not None
            ]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def test_concurrent_replacements_serialize_native_cancellation(self):
        started = {name: asyncio.Event() for name in ("user-1", "user-2", "user-3")}
        transport_tasks = {}
        draining, release_drain = asyncio.Event(), asyncio.Event()
        drain_count = 0

        async def stream(request):
            transport_tasks[request.id] = asyncio.current_task()
            started[request.id].set()
            yield {"type": "agent_status", "status": "started"}
            await asyncio.Event().wait()

        namespace, service = _namespace_for_stream(stream)

        async def wait_for_session(session_id):
            nonlocal drain_count
            drain_count += 1
            if drain_count == 1:
                draining.set()
                await release_drain.wait()

        service.wait_for_session.side_effect = wait_for_session
        handlers = []
        try:
            async with asyncio.timeout(5):
                await namespace.on_assistant_message("sid-1", _message("user-1"))
                await started["user-1"].wait()
                handlers.append(
                    asyncio.create_task(namespace.on_assistant_message("sid-1", _message("user-2")))
                )
                await draining.wait()
                handlers.append(
                    asyncio.create_task(namespace.on_assistant_message("sid-1", _message("user-3")))
                )
                await asyncio.sleep(0)
                assert service.cancel_session.await_count == 1
                release_drain.set()
                await asyncio.gather(*handlers)
                await started["user-3"].wait()
                assert service.cancel_session.await_count == 2
                assert service.wait_for_session.await_count == 2
                assert namespace._active_streams["sess-1"] is transport_tasks["user-3"]
                assert transport_tasks["user-1"].done()
        finally:
            tasks = [*handlers, *transport_tasks.values(), *namespace._active_streams.values()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @pytest.mark.parametrize("ending", ["completed", "deferred", "failure"])
    async def test_old_stream_release_keeps_new_owner(self, ending):
        started, finish = asyncio.Event(), asyncio.Event()

        async def stream(_request):
            started.set()
            await finish.wait()
            if ending == "failure":
                raise RuntimeError("Old transport failed")
            if ending == "deferred":
                yield {"type": "final_response", "pending_tool_call": {"tool_call_id": "call-1"}}
            yield {"type": "agent_status", "status": "completed"}

        namespace, _service = _namespace_for_stream(stream)
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        old_task = namespace._active_streams["sess-1"]
        replacement = asyncio.create_task(_never_finishes())
        try:
            async with asyncio.timeout(5):
                await started.wait()
                namespace._active_streams["sess-1"] = replacement
                finish.set()
                await old_task
                assert namespace._active_streams["sess-1"] is replacement
        finally:
            old_task.cancel()
            replacement.cancel()
            await asyncio.gather(old_task, replacement, return_exceptions=True)

    async def test_disconnect_allows_turn_to_finish(self):
        started, finish, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def stream(_request):
            try:
                started.set()
                yield {"type": "text_delta", "content": "Partial"}
                await finish.wait()
                yield {"type": "agent_status", "status": "completed"}
            finally:
                closed.set()

        namespace, service = _namespace_for_stream(stream)
        await namespace.on_assistant_message("sid-1", _message("user-1"))
        task = namespace._active_streams["sess-1"]
        try:
            async with asyncio.timeout(5):
                await started.wait()
                await namespace.on_disconnect("sid-1")
                assert task.cancelling() == 0
                service.cancel_session.assert_not_awaited()
                finish.set()
                await task
                assert closed.is_set()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_emit_failure_closes_service_stream_before_task_finishes(self):
        closed = asyncio.Event()

        async def stream(_request):
            try:
                yield {"type": "text_delta", "content": "Partial"}
                await asyncio.Event().wait()
            finally:
                closed.set()

        namespace, _service = _namespace_for_stream(stream)
        namespace.emit.side_effect = RuntimeError("Socket write failed")
        await namespace.on_assistant_message("sid-1", _message("user-1"))

        await namespace._active_streams["sess-1"]

        assert closed.is_set()
        assert namespace._active_streams == {}
