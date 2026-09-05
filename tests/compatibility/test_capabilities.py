"""Native lifecycle and event capabilities retain their execution semantics."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ProcessEventStream
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.main import AssistantDefinition, create_runtime
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, request


async def test_native_node_guard_blocks_model_request_and_streamed_text(
    isolated_services, monkeypatch
):
    observed = []

    class BlockModel(AbstractCapability):
        async def before_node_run(self, ctx, *, node):
            if Agent.is_model_request_node(node):
                observed.append("guard")
                raise ValueError("Model request prohibited by host")
            return node

    async def respond(messages, info):
        observed.append("model")
        yield "This response must never be generated"

    model = FunctionModel(stream_function=respond)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(capabilities=[BlockModel()])
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = [event async for event in runtime.stream_message(request())]
    assert_terminal(events, error=True)
    assert observed == ["guard"]
    assert not any(event["type"] == "text_delta" for event in events)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "model-error"])
async def test_native_node_wrapper_encloses_model_request_and_cleans_up(
    isolated_services, monkeypatch, fail
):
    observed = []

    class ModelScope(AbstractCapability):
        async def wrap_node_run(self, ctx, *, node, handler):
            if not Agent.is_model_request_node(node):
                return await handler(node)
            observed.append("enter")
            try:
                result = await handler(node)
                observed.append("success")
                return result
            except RuntimeError:
                observed.append("error")
                raise
            finally:
                observed.append("exit")

    async def respond(messages, info):
        observed.append("model")
        if fail:
            raise RuntimeError("Model unavailable")
        yield "Ready."

    model = FunctionModel(stream_function=respond)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(capabilities=[ModelScope()])
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = [event async for event in runtime.stream_message(request())]
    assert_terminal(events, error=fail)
    assert observed == ["enter", "model", "error" if fail else "success", "exit"]
    if not fail:
        assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Ready."


async def test_native_event_processor_shapes_wire_events_without_changing_execution(
    isolated_services, monkeypatch
):
    called_with = []
    model_returns = []
    native_outputs = []

    class ObserveResult(AbstractCapability):
        async def after_run(self, ctx, *, result):
            native_outputs.append(result.output)
            return result

    def lookup(item: str) -> str:
        called_with.append(item)
        return "original tool output"

    async def respond(messages, info):
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield "Starting "
            yield {0: DeltaToolCall("lookup", '{"item":"lamp"}', tool_call_id="lookup-1")}
        else:
            model_returns.extend(part.content for part in returns)
            yield "Final "
            yield "answer"

    async def presentation(ctx, stream):
        async for event in stream:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                event = replace(event, part=replace(event.part, content=event.part.content.upper()))
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                event = replace(
                    event,
                    delta=replace(event.delta, content_delta=event.delta.content_delta.upper()),
                )
            elif isinstance(event, FunctionToolCallEvent):
                event = replace(event, part=replace(event.part, args={"item": "display item"}))
            elif isinstance(event, FunctionToolResultEvent):
                event = replace(event, part=replace(event.part, content="display result"))
            yield event

    model = FunctionModel(stream_function=respond)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(
        tools=[lookup], capabilities=[ProcessEventStream(presentation), ObserveResult()]
    )
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = [event async for event in runtime.stream_message(request())]
    assert_terminal(events)
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == (
        "STARTING FINAL ANSWER"
    )
    tool_call = next(e for e in events if e["type"] == "tool_call")
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_call["arguments"] == {"item": "display item"}
    assert tool_result["output"] == "display result"
    assert called_with == ["lamp"]
    assert model_returns == ["original tool output"]
    assert native_outputs == ["Final answer"]


async def test_native_tool_failure_is_reported_and_preserved_on_next_turn(
    isolated_services, monkeypatch
):
    outcomes = []

    def lookup() -> str:
        raise ToolFailed("Catalog unavailable")

    async def respond(messages, info):
        returns = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield {0: DeltaToolCall("lookup", "{}", tool_call_id="lookup-1")}
        else:
            outcomes.extend(part.outcome for part in returns)
            yield "The catalog is unavailable."

    model = FunctionModel(stream_function=respond)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    definition = AssistantDefinition(tools=[lookup])
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        events = [event async for event in runtime.stream_message(request())]
        first = assert_terminal(events)
        assert_terminal(
            [
                event
                async for event in runtime.stream_message(
                    request(id="user-2", parent_id=first["message_id"], content="Try later")
                )
            ]
        )
    assert outcomes == ["failed", "failed"]
    errors = [event for event in events if event["type"] == "tool_error"]
    assert len(errors) == 1
    assert errors[0]["call_id"] == "lookup-1"
    assert errors[0]["error"] == "Catalog unavailable"
