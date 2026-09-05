"""Socket cancellation closes the lifecycle even before native events exist."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from assistant_runtime.app.socketio_server import AssistantNamespace

from .test_execution import assert_terminal, request
from .test_runtime_cancellation import drain


async def test_socket_cancellation_during_planning_drains_and_emits_terminal_envelope(
    runtime, script, monkeypatch
):
    planning, closed = asyncio.Event(), asyncio.Event()

    async def lookup(session_id):
        planning.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(runtime.sessions, "get_context_if_exists_async", lookup)
    namespace = AssistantNamespace("/assistant")
    namespace.server = SimpleNamespace(
        fastapi_app=SimpleNamespace(state=SimpleNamespace(streaming_service=runtime.streaming))
    )
    namespace.emit = AsyncMock()
    await namespace.on_assistant_message("client-1", request().model_dump())
    task = namespace._active_streams["compat"]
    try:
        async with asyncio.timeout(5):
            await planning.wait()
            assert namespace.emit.await_count == 0
            await namespace.on_assistant_cancel("client-1", {"session_id": "compat"})
            await namespace._active_streams["compat"]
            assert closed.is_set()
            assert task.done()
            assert await runtime.streaming.cancel_session("compat") is False
    finally:
        await drain(task)

    events = [call.args[1] for call in namespace.emit.await_args_list]
    final = assert_terminal(events, error=True)
    assert final["error_type"] == "cancelled"
    assert final["session_id"] == "compat"
    assert "message_id" not in final
    assert len(events) == 4
    assert events[2]["type"] == "error"
    assert events[2]["error_type"] == "cancelled"
    assert events[2]["terminal"] is True
    assert all(call.kwargs["to"] == "client-1" for call in namespace.emit.await_args_list)
    assert script.requests == []
    assert namespace._active_streams == {}
    assert namespace._stream_delivery == {}
