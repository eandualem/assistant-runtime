"""Cancellation boundaries preserve native history and serialize session writes."""

from __future__ import annotations

import asyncio

from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
from pydantic_ai.models.function import FunctionModel

from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, request
from .test_runtime_cancellation import collect, drain


async def test_cancellation_does_not_expose_text_suppressed_by_event_middleware(
    isolated_services, monkeypatch
):
    suppressed = asyncio.Event()

    async def response(messages, info):
        yield "Private native response"
        await asyncio.Event().wait()

    async def redact(ctx, stream):
        async for event in stream:
            if (isinstance(event, PartStartEvent) and isinstance(event.part, TextPart)) or (
                isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta)
            ):
                suppressed.set()
                continue
            yield event

    model = FunctionModel(stream_function=response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(capabilities=[ProcessEventStream(redact)])
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = []
        task = asyncio.create_task(collect(runtime.stream_message(request()), events))
        try:
            async with asyncio.timeout(5):
                await suppressed.wait()
                assert await runtime.cancel_session("compat")
                await task
        finally:
            await drain(task)
        assert assert_terminal(events, error=True)["content"] == ""
        assert "Private native response" not in str(events)
        sessions = runtime._assistant_service.get_session_store()
        assert (await sessions.get_message_path("compat"))[-1][
            "content"
        ] == "Private native response"


async def test_timeout_persists_native_partial_response(runtime, script, monkeypatch):
    closed = asyncio.Event()

    async def response(messages, info):
        try:
            yield "Before timeout"
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    runtime.streaming._config = runtime.streaming._config.model_copy(
        update={"stream_timeout_seconds": 0.05}
    )
    async with asyncio.timeout(5):
        events = [event async for event in runtime.streaming.stream_message(request())]
    assert closed.is_set()
    assert assert_terminal(events, error=True)["error_type"] == "timeout"
    assert (await runtime.sessions.get_message_path("compat"))[-1]["content"] == "Before timeout"


async def test_replacement_waits_for_prior_snapshot_before_validating_its_parent(
    runtime, script, monkeypatch
):
    received, saving, release_save = asyncio.Event(), asyncio.Event(), asyncio.Event()
    second_started = asyncio.Event()
    calls = 0

    async def response(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield "First partial"
            await asyncio.Event().wait()
        else:
            second_started.set()
            yield "Next answer"

    original_save = runtime.sessions.register_assistant_message

    async def save(session_id, **kwargs):
        if kwargs["content"] == "First partial":
            saving.set()
            await release_save.wait()
        return await original_save(session_id, **kwargs)

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    monkeypatch.setattr(runtime.sessions, "register_assistant_message", save)
    first_events, second_events = [], []
    first = asyncio.create_task(
        collect(runtime.streaming.stream_message(request()), first_events, text_received=received)
    )
    second = None
    try:
        async with asyncio.timeout(5):
            await received.wait()
            assistant_id = runtime.sessions.get_context("compat")["current_assistant_message_id"]
            second = asyncio.create_task(
                collect(
                    runtime.streaming.stream_message(
                        request(id="user-2", parent_id=assistant_id, content="Continue")
                    ),
                    second_events,
                )
            )
            await saving.wait()
            assert not second_started.is_set()
            assert "user-2" not in runtime.sessions.get_context("compat")["message_index"]
            release_save.set()
            await asyncio.gather(first, second)
    finally:
        release_save.set()
        await drain(first)
        if second is not None:
            await drain(second)
    assert assert_terminal(first_events, error=True)["error_type"] == "cancelled"
    assert_terminal(second_events)
    path = await runtime.sessions.get_message_path("compat")
    assert [message["content"] for message in path] == [
        "Help",
        "First partial",
        "Continue",
        "Next answer",
    ]
    assert not runtime.sessions.get_context("compat").get("current_assistant_message_id")
