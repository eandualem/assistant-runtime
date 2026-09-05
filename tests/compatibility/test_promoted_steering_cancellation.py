"""Idle steering remains queued until native execution consumes it."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import FunctionModel

from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, request


@pytest.mark.parametrize("accept_first", [True, False], ids=["accepted", "direct"])
@pytest.mark.parametrize("cancel", [True, False], ids=["cancelled-setup", "completed"])
async def test_promoted_steering_is_acknowledged_only_after_consumption(
    isolated_services, monkeypatch, accept_first, cancel
):
    setup_started, setup_closed = asyncio.Event(), asyncio.Event()
    observed_steering = []

    async def dependencies(request):
        if cancel and request.is_steering:
            setup_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                setup_closed.set()

    async def response(messages, info):
        observed_steering.append(
            [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and "Use the blue item" in part.content
            ]
        )
        yield "Adjusted" if observed_steering[-1] else "Initial response"

    model = FunctionModel(stream_function=response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(deps_factory=dependencies)
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        initial = await runtime.run_message(request())
        steering = request(id="steering-1", message_type="steering", content="Use the blue item")
        if accept_first:
            assert await runtime.accept_steering(steering, has_live_stream=False) == "promoted"

        async def collect():
            return [event async for event in runtime.stream_message(steering)]

        task = asyncio.create_task(collect())
        try:
            async with asyncio.timeout(5):
                if cancel:
                    await setup_started.wait()
                    assert await runtime.cancel_session("compat") is True
                events = await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        context = runtime._assistant_service.get_session_store().get_context("compat")
        final = assert_terminal(events, error=cancel)
        assert final["message_id"] == initial.message_id
        if cancel:
            assert setup_closed.is_set()
            assert final["error_type"] == "cancelled"
            assert context["pending_steering_ids"] == ["steering-1"]
            assert context["steering_index"]["steering-1"]["status"] == "pending"
            resumed = await runtime.run_message(
                request(id="user-2", parent_id=initial.message_id, content="Continue")
            )
            assert resumed.content == "Adjusted"
        assert len(observed_steering) == 2
        assert observed_steering[0] == []
        assert len(observed_steering[1]) == 1
        assert context["pending_steering_ids"] == []
        assert context["steering_index"]["steering-1"]["status"] == "delivered"


async def test_concurrent_idle_steering_does_not_run_an_already_consumed_request(
    isolated_services, monkeypatch
):
    model_requests = []

    async def response(messages, info):
        model_requests.append(
            [
                part.content
                for message in messages
                for part in message.parts
                if isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and "Additional user steering" in part.content
            ]
        )
        yield "Applied both" if model_requests[-1] else "Initial response"

    model = FunctionModel(stream_function=response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    async with create_runtime(settings=isolated_services) as runtime:
        initial = await runtime.run_message(request())
        steering = [
            request(id=f"steering-{i}", message_type="steering", content=f"Change {i}")
            for i in range(2)
        ]
        for pending in steering:
            assert await runtime.accept_steering(pending, has_live_stream=False) == "promoted"

        async def collect(pending):
            return [event async for event in runtime.stream_message(pending)]

        async with asyncio.timeout(5):
            first, second = await asyncio.gather(*(collect(pending) for pending in steering))

        completed = assert_terminal(first)
        assert completed["message_id"] == initial.message_id
        rejected = assert_terminal(second, error=True)
        assert rejected["error_type"] == "session_error"
        errors = [event for event in second if event["type"] == "error"]
        assert len(errors) == 1
        assert "already delivered by another turn" in errors[0]["message"]
        assert model_requests == [
            [],
            [f"Additional user steering while you were working:\nChange {i}" for i in range(2)],
        ]
        context = runtime._assistant_service.get_session_store().get_context("compat")
        assert context["pending_steering_ids"] == []
        assert all(context["steering_index"][r.id]["status"] == "delivered" for r in steering)
        assert not context.get("current_assistant_message_id")
