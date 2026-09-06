"""Native cancellation saves accepted work independently of stream consumers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools._request_context import record_current_telegram_chat_binding

from .test_execution import assert_terminal, calls, register_lookup, request


async def collect(stream, events, *, text_received=None, tool_finished=None):
    async for event in stream:
        events.append(event)
        if text_received is not None and event["type"] == "text_delta":
            text_received.set()
        if (
            tool_finished is not None
            and event["type"] == "tool_result"
            and event["call_id"] == "done-1"
        ):
            tool_finished.set()


async def drain(task):
    if not task.done():
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def assert_cancelled(events, assistant_id):
    final = assert_terminal(events, error=True)
    assert final["error_type"] == "cancelled"
    assert final["message_id"] == assistant_id
    errors = [event for event in events if event["type"] == "error"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "cancelled"
    assert errors[0]["terminal"] is True
    assert errors[0]["retry_allowed"] is False
    return final


async def test_explicit_cancel_saves_partial_native_response(runtime, script, monkeypatch):
    received, closed = asyncio.Event(), asyncio.Event()

    async def response(messages, info):
        try:
            yield "Partial response"
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    events = []
    task = asyncio.create_task(
        collect(runtime.streaming.stream_message(request()), events, text_received=received)
    )
    try:
        async with asyncio.timeout(5):
            await received.wait()
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            assert await runtime.streaming.cancel_session("compat") is True
            await runtime.streaming.wait_for_session("compat")
            await task
    finally:
        await drain(task)

    assert closed.is_set()
    assert_cancelled(events, assistant_id)
    path = await runtime.sessions.get_message_path("compat")
    assert [message["role"] for message in path] == ["user", "assistant"]
    assert path[-1]["id"] == assistant_id
    assert path[-1]["content"] == "Partial response"
    assert await runtime.streaming.cancel_session("compat") is False
    assert not runtime.sessions.get_context("compat").get("current_assistant_message_id")
    assert any(
        isinstance(part, TextPart) and part.content == "Partial response"
        for message in runtime.sessions.get_history("compat")
        for part in message.parts
    )


async def test_setup_cancel_drains_dependency_factory_without_starting_model(
    isolated_services, monkeypatch
):
    started, closed = asyncio.Event(), asyncio.Event()
    model_calls = []

    async def dependencies(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    async def response(messages, info):
        model_calls.append(messages)
        yield "Should not run"

    model = FunctionModel(stream_function=response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(deps_factory=dependencies)
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = []
        task = asyncio.create_task(collect(runtime.stream_message(request()), events))
        try:
            async with asyncio.timeout(5):
                await started.wait()
                sessions = runtime._assistant_service.get_session_store()
                assistant_id = sessions.get_context("compat")["current_assistant_message_id"]
                assert await runtime.cancel_session("compat") is True
                await task
        finally:
            await drain(task)
        assert closed.is_set()
        assert model_calls == []
        assert_cancelled(events, assistant_id)
        path = await sessions.get_message_path("compat")
        assert [message["role"] for message in path] == ["user", "assistant"]
        assert path[-1]["id"] == assistant_id
        assert path[-1]["content"] == ""
        assert not sessions.get_context("compat").get("current_assistant_message_id")


async def test_planning_cancel_drains_session_lookup_before_accepting_a_message(
    runtime, script, monkeypatch
):
    started, closed = asyncio.Event(), asyncio.Event()

    async def lookup(session_id):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(runtime.sessions, "get_context_if_exists_async", lookup)
    events = []
    task = asyncio.create_task(collect(runtime.streaming.stream_message(request()), events))
    try:
        async with asyncio.timeout(5):
            await started.wait()
            assert await runtime.streaming.cancel_session("compat") is True
            await runtime.streaming.wait_for_session("compat")
            await task
    finally:
        await drain(task)
    assert closed.is_set()
    final = assert_terminal(events, error=True)
    assert final["error_type"] == "cancelled"
    assert final.get("message_id") is None
    assert not runtime.sessions.has_session("compat")
    assert script.requests == []


async def test_ingress_cancel_drains_before_model_setup_and_saves_accepted_message(runtime, script):
    started, closed = asyncio.Event(), asyncio.Event()

    async def ingress_drain(session_id):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    runtime.streaming.attach_ingress(SimpleNamespace(drain=ingress_drain))
    events = []
    task = asyncio.create_task(collect(runtime.streaming.stream_message(request()), events))
    try:
        async with asyncio.timeout(5):
            await started.wait()
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            assert await runtime.streaming.cancel_session("compat") is True
            await runtime.streaming.wait_for_session("compat")
            await task
    finally:
        await drain(task)
    assert closed.is_set()
    assert_cancelled(events, assistant_id)
    path = await runtime.sessions.get_message_path("compat")
    assert [message["role"] for message in path] == ["user", "assistant"]
    assert path[-1]["id"] == assistant_id
    assert path[-1]["content"] == ""
    assert script.requests == []


async def test_metadata_save_failure_cannot_replace_cancellation_or_discard_work(
    runtime, script, monkeypatch
):
    received, closed = asyncio.Event(), asyncio.Event()

    async def bind_chat(item):
        record_current_telegram_chat_binding("chat-1")
        return {"bound": True}

    async def response(messages, info):
        if not any(
            isinstance(part, ToolReturnPart) for message in messages for part in message.parts
        ):
            yield calls(("lookup", '{"item":"sample"}', "bind-1"))
        else:
            try:
                yield "Partial response after binding"
                await asyncio.Event().wait()
            finally:
                closed.set()

    register_lookup(runtime, bind_chat)
    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    events = []
    task = asyncio.create_task(
        collect(runtime.streaming.stream_message(request()), events, text_received=received)
    )
    try:
        async with asyncio.timeout(5):
            await received.wait()
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            failed_save = AsyncMock(side_effect=RuntimeError("Metadata store unavailable"))
            monkeypatch.setattr(runtime.sessions, "save_session_state_async", failed_save)
            assert await runtime.streaming.cancel_session("compat") is True
            await task
    finally:
        await drain(task)
    failed_save.assert_awaited_once_with("compat")
    assert closed.is_set()
    assert_cancelled(events, assistant_id)
    path = await runtime.sessions.get_message_path("compat")
    assert path[-1]["id"] == assistant_id
    assert path[-1]["content"] == "Partial response after binding"
    returns = [
        part
        for message in runtime.sessions.get_history("compat")
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(returns) == 1
    assert returns[0].tool_call_id == "bind-1"
    assert returns[0].content == {"bound": True}
    assert returns[0].outcome == "success"


async def test_cancel_keeps_completed_tool_and_marks_inflight_call_interrupted(runtime, script):
    started, closed, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    executed = []

    async def lookup(item):
        executed.append(item)
        if item == "done":
            return {"value": "ready"}
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    register_lookup(runtime, lookup)
    script.steps = [
        [
            calls(
                ("lookup", '{"item":"done"}', "done-1"),
                ("lookup", '{"item":"inflight"}', "inflight-1"),
            )
        ]
    ]
    events = []
    task = asyncio.create_task(
        collect(runtime.streaming.stream_message(request()), events, tool_finished=completed)
    )
    try:
        async with asyncio.timeout(5):
            await started.wait()
            await completed.wait()
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            assert await runtime.streaming.cancel_session("compat") is True
            await task
    finally:
        await drain(task)
    assert closed.is_set()
    assert sorted(executed) == ["done", "inflight"]
    assert len(script.requests) == 1
    assert_cancelled(events, assistant_id)
    parts = [part for message in runtime.sessions.get_history("compat") for part in message.parts]
    assert {part.tool_call_id for part in parts if isinstance(part, ToolCallPart)} == {
        "done-1",
        "inflight-1",
    }
    returns = {part.tool_call_id: part for part in parts if isinstance(part, ToolReturnPart)}
    assert returns["done-1"].content == {"value": "ready"}
    assert returns["done-1"].outcome == "success"
    assert returns["inflight-1"].outcome == "interrupted"
    assert "unknown" in str(returns["inflight-1"].content)
    context = runtime.sessions.get_context("compat")
    stored = {
        tool["id"]: tool
        for segment in context["message_index"][assistant_id]["segments"]
        if segment["kind"] == "tool_group"
        for tool in segment["tools"]
    }
    assert "status" not in stored["done-1"]
    assert stored["inflight-1"]["status"] == "cancelled"
    assert not context.get("pending_tool_call_id")
    assert not context.get("current_assistant_message_id")


async def test_continuation_cancel_during_setup_keeps_accepted_host_result(
    isolated_services, host_schema, script, monkeypatch
):
    started, closed = asyncio.Event(), asyncio.Event()
    resolved = []

    async def dependencies(request):
        resolved.append(request.id)
        if request.is_continuation:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

    script.steps = [[calls(("select_item", '{"item":"sample"}', "host-1"))]]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    settings = isolated_services.model_copy(
        update={"tools": isolated_services.tools.model_copy(update={"host_tools": host_schema})}
    )
    definition = AssistantDefinition(deps_factory=dependencies)
    async with create_runtime(assistant=definition, settings=settings) as runtime:
        first = assert_terminal([event async for event in runtime.stream_message(request())])
        assistant_id = first["message_id"]
        assert first["pending_tool_call"]["call_id"] == "host-1"
        sessions = runtime._assistant_service.get_session_store()
        events = []
        task = asyncio.create_task(
            collect(
                runtime.stream_message(
                    request(
                        id="continuation-1",
                        parent_id=assistant_id,
                        content="",
                        tool_call_id="host-1",
                        tool_result={"selected": True},
                    )
                ),
                events,
            )
        )
        try:
            async with asyncio.timeout(5):
                await started.wait()
                assert await runtime.cancel_session("compat") is True
                await task
        finally:
            await drain(task)
        assert closed.is_set()
        assert_cancelled(events, assistant_id)
        assert resolved == ["user-1", "continuation-1"]
        assert len(script.requests) == 1
        path = await sessions.get_message_path("compat")
        assert len(path) == 2
        assert path[-1]["id"] == assistant_id
        returns = [
            part
            for message in sessions.get_history("compat")
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(returns) == 1
        assert returns[0].tool_call_id == "host-1"
        assert returns[0].content == {"selected": True}
        assert returns[0].outcome == "success"
        assert not sessions.get_context("compat").get("pending_tool_call_id")


@pytest.mark.parametrize("action", ["cancel", "close", "stop"])
async def test_paused_consumer_does_not_block_cancellation_persistence(
    action, runtime, script, monkeypatch
):
    closed = asyncio.Event()

    async def response(messages, info):
        try:
            yield "Paused consumer response"
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    stream = runtime.streaming.stream_message(request())
    events = []
    try:
        async with asyncio.timeout(5):
            async for event in stream:
                events.append(event)
                if event["type"] == "text_delta":
                    break
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            # No further event pulls occur until the producer has fully finalized.
            if action == "close":
                await stream.aclose()
            elif action == "stop":
                await runtime.streaming.stop()
            else:
                assert await runtime.streaming.cancel_session("compat") is True
                await runtime.streaming.wait_for_session("compat")
            assert closed.is_set()
            path = await runtime.sessions.get_message_path("compat")
            assert path[-1]["id"] == assistant_id
            assert path[-1]["content"] == "Paused consumer response"
            assert not runtime.sessions.get_context("compat").get("current_assistant_message_id")
            if action != "close":
                events.extend([event async for event in stream])
                assert_cancelled(events, assistant_id)
    finally:
        await stream.aclose()
