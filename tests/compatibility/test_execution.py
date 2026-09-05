"""Application contract versus public upstream execution; no Agent mocks."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    BinaryContent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessagesTypeAdapter,
    PartStartEvent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall, FunctionModel
from pydantic_ai.run import AgentRunResultEvent

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.services.tools._host_tools import build_host_toolset
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


def request(**overrides):
    return AssistantRequest.model_validate(
        {"id": "user-1", "session_id": "compat", "parent_id": None, "content": "Help", **overrides}
    )


def calls(*items):
    return {
        index: DeltaToolCall(name=name, json_args=args, tool_call_id=call_id)
        for index, (name, args, call_id) in enumerate(items)
    }


def assert_terminal(events, *, error=False):
    assert events[0]["type"] == "agent_status"
    assert events[0]["status"] == "started"
    assert events[-1]["type"] == "agent_status"
    assert events[-1]["status"] == "completed"
    assert sum(e.get("status") == "started" for e in events) == 1
    assert sum(e.get("status") == "completed" for e in events) == 1
    finals = [e for e in events if e["type"] == "final_response"]
    assert len(finals) == 1
    assert bool(finals[0].get("error")) == error
    assert any(e["type"] == "error" for e in events) == error
    return finals[0]


def register_lookup(runtime, handler):
    runtime.tools.register_backend_tool(
        ToolDefinition(
            name="lookup",
            description="Look up an item.",
            parameters_schema={
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
            category=ToolCategory.BACKEND,
        ),
        handler,
    )


async def test_runtime_stream_order_and_complete_tool_arguments(runtime, script):
    script.steps = [
        [
            {0: DeltaThinkingPart(content="Checking")},
            {0: DeltaThinkingPart(content=" the item")},
            "Let me check. ",
            {1: DeltaToolCall(name="lookup", json_args='{"item":', tool_call_id="lookup-1")},
            {1: DeltaToolCall(json_args='"sample"}')},
        ],
        ["Found ", "it."],
    ]
    executed = []

    async def lookup(item):
        executed.append(item)
        return {"value": "ready"}

    register_lookup(runtime, lookup)
    events = [e async for e in runtime.streaming.stream_message(request())]
    assert_terminal(events)
    types = [e["type"] for e in events]
    assert types.index("thinking_delta") < types.index("text_delta") < types.index("tool_call")
    assert types.index("tool_call") < types.index("tool_result")
    assert types.index("tool_result") < len(types) - 1 - types[::-1].index("text_delta")
    assert (
        "".join(e["content"] for e in events if e["type"] == "thinking_delta")
        == "Checking the item"
    )
    assert (
        "".join(e["content"] for e in events if e["type"] == "text_delta")
        == "Let me check. Found it."
    )
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["arguments"] == {"item": "sample"}
    assert executed == ["sample"]
    path = await runtime.sessions.get_message_path("compat")
    assert [m["role"] for m in path] == ["user", "assistant"]
    assert [s["kind"] for s in path[-1]["segments"]] == ["thinking", "text", "tool_group", "text"]


async def test_public_stream_has_complete_tools_and_one_trailing_result(script):
    script.steps = [
        [
            {9: DeltaThinkingPart(content="Checking")},
            "Looking. ",
            calls(("lookup", '{"item":"sample"}', "lookup-1")),
        ],
        ["Found it."],
    ]
    agent = Agent(script.model())

    @agent.tool_plain
    def lookup(item: str) -> str:
        return f"Found {item}"

    async with agent.run_stream_events("Help") as stream:
        events = [e async for e in stream]
    tool_calls = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    results = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    assert len(tool_calls) == len(results) == 1
    assert tool_calls[0].part.args_as_dict() == {"item": "sample"}
    assert results[0].part.tool_call_id == tool_calls[0].part.tool_call_id == "lookup-1"
    assert results[0].part.content == "Found sample"
    assert events.index(tool_calls[0]) < events.index(results[0]) < len(events) - 1
    assert sum(isinstance(e, AgentRunResultEvent) for e in events) == 1
    assert isinstance(events[-1], AgentRunResultEvent)
    assert events[-1].result.output == "Found it."


async def test_runtime_host_continuation_preserves_message_and_does_not_reexecute_backend(
    runtime, script
):
    script.steps = [
        [
            calls(
                ("lookup", '{"item":"sample"}', "lookup-1"),
                ("select_item", '{"item":"sample"}', "host-1"),
            )
        ],
        ["Selected."],
    ]
    executed = []

    async def lookup(item):
        executed.append(item)
        return {"value": item}

    register_lookup(runtime, lookup)
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    assert first["pending_tool_call"] == {
        "tool_name": "select_item",
        "call_id": "host-1",
        "arguments": {"item": "sample"},
    }
    continuation = request(
        id="continuation-1",
        parent_id="user-1",
        content="",
        tool_call_id="host-1",
        tool_result={"selected": True},
    )
    second_events = [e async for e in runtime.streaming.stream_message(continuation)]
    second = assert_terminal(second_events)
    assert second["message_id"] == first["message_id"]
    assert second["content"] == ""
    assert second["streamed"] is True
    assert "".join(e["content"] for e in second_events if e["type"] == "text_delta") == "Selected."
    assert not any(e["type"] == "tool_call" for e in second_events)
    assert executed == ["sample"]
    assert len(await runtime.sessions.get_message_path("compat")) == 2
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [p.tool_call_id for p in returns] == ["lookup-1", "host-1"]
    assert not runtime.sessions.get_context("compat").get("pending_tool_call_id")


async def test_runtime_rejects_multiple_deferred_host_calls(runtime, script):
    script.steps = [
        [
            calls(
                ("select_item", '{"item":"one"}', "host-1"),
                ("select_item", '{"item":"two"}', "host-2"),
            )
        ]
    ]
    events = [e async for e in runtime.streaming.stream_message(request())]
    assert_terminal(events, error=True)
    assert any(
        "expected exactly 1 deferred tool call, got 2" in e.get("message", "") for e in events
    )
    assert not runtime.sessions.get_context("compat").get("pending_tool_call_id")


async def test_public_deferred_calls_round_trip_as_a_batch(script, host_schema):
    script.steps = [
        [
            calls(
                ("select_item", '{"item":"one"}', "host-1"),
                ("select_item", '{"item":"two"}', "host-2"),
            )
        ],
        ["Both selected."],
    ]
    agent = Agent(
        script.model(),
        toolsets=[build_host_toolset(host_schema)],
        output_type=[str, DeferredToolRequests],
    )
    async with agent.run_stream_events("Select both") as stream:
        first_events = [e async for e in stream]
    first = first_events[-1].result
    assert isinstance(first.output, DeferredToolRequests)
    assert [c.tool_call_id for c in first.output.calls] == ["host-1", "host-2"]
    restored = ModelMessagesTypeAdapter.validate_json(first.all_messages_json())
    async with agent.run_stream_events(
        message_history=restored,
        deferred_tool_results=DeferredToolResults(
            calls={"host-1": "one selected", "host-2": "two selected"}
        ),
    ) as stream:
        second_events = [e async for e in stream]
    assert second_events[-1].result.output == "Both selected."
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert {p.tool_call_id: p.content for p in returns} == {
        "host-1": "one selected",
        "host-2": "two selected",
    }


async def test_runtime_delivers_mid_tool_steering_once(runtime, script):
    script.steps = [[calls(("lookup", '{"item":"sample"}', "lookup-1"))], ["Adjusted."]]

    async def lookup(item):
        status = await runtime.streaming.accept_steering(
            request(id="steering-1", content="Use the blue item", message_type="steering"),
            has_live_stream=True,
        )
        assert status == "queued"
        return {"value": item}

    register_lookup(runtime, lookup)
    assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    prompts = [
        p.content for m in script.requests[-1] for p in m.parts if isinstance(p, UserPromptPart)
    ]
    assert sum("Use the blue item" in p for p in prompts) == 1
    context = runtime.sessions.get_context("compat")
    assert context["pending_steering_ids"] == []
    assert context["steering_index"]["steering-1"]["status"] == "delivered"
    assert len(context["message_index"]) == 2


async def test_public_enqueue_delivers_before_next_model_request(script):
    script.steps = [[calls(("lookup", "{}", "lookup-1"))], ["Adjusted."]]
    agent = Agent(script.model())
    started, release = asyncio.Event(), asyncio.Event()

    @agent.tool_plain
    async def lookup() -> str:
        started.set()
        await release.wait()
        return "ready"

    async def steer(run):
        await started.wait()
        run.enqueue("Use the blue item", priority="asap")
        release.set()

    async with asyncio.timeout(5), agent.iter("Help") as run, asyncio.TaskGroup() as tasks:
        tasks.create_task(steer(run))
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for _ in stream:
                        pass
    assert run.result.output == "Adjusted."
    prompts = [
        p.content for m in script.requests[-1] for p in m.parts if isinstance(p, UserPromptPart)
    ]
    assert prompts.count("Use the blue item") == 1


async def test_public_cancellation_preserves_partial_response_and_closes_stream():
    closed = asyncio.Event()

    async def slow_response(messages, info):
        try:
            yield "Partial response"
            await asyncio.Event().wait()
        finally:
            closed.set()

    agent = Agent(FunctionModel(stream_function=slow_response))
    events = []

    async def consume():
        async with asyncio.timeout(5), agent.run_stream_events("Help") as stream:
            async for event in stream:
                events.append(event)
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    stream.cancel()

    with pytest.raises(RunCancelled) as caught:
        await consume()
    assert closed.is_set()
    assert not any(isinstance(e, AgentRunResultEvent) for e in events)
    assert any(
        isinstance(p, TextPart) and p.content == "Partial response"
        for m in caught.value.all_messages()
        for p in m.parts
    )


async def test_runtime_task_cancellation_persists_partial_turn_and_preserves_caller_cancellation(
    runtime, script, monkeypatch
):
    """A cancelled consumer still waits for native snapshot persistence."""
    received, closed = asyncio.Event(), asyncio.Event()

    async def slow_response(messages, info):
        try:
            yield "Partial response"
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=slow_response))
    events = []

    async def collect():
        async for event in runtime.streaming.stream_message(request()):
            events.append(event)
            if event["type"] == "text_delta":
                received.set()

    task = asyncio.create_task(collect())
    try:
        async with asyncio.timeout(5):
            await received.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert closed.is_set()
    assert not runtime.sessions.get_context("compat").get("current_assistant_message_id")
    path = await runtime.sessions.get_message_path("compat")
    assert [m["role"] for m in path] == ["user", "assistant"]
    assert path[-1]["content"] == "Partial response"
    assert await runtime.streaming.cancel_session("compat") is False


async def test_public_cancel_drains_inflight_tool_and_retains_call_history(script):
    script.steps = [[calls(("lookup", "{}", "lookup-1"))]]
    agent = Agent(script.model())
    started, closed = asyncio.Event(), asyncio.Event()

    @agent.tool_plain
    async def lookup() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()
        return "unreachable"

    async def cancel(run):
        await started.wait()
        run.cancel()

    async def consume():
        async with asyncio.timeout(5), agent.iter("Help") as run, asyncio.TaskGroup() as tasks:
            tasks.create_task(cancel(run))
            async for node in run:
                if Agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream:
                            pass

    with pytest.raises(RunCancelled) as caught:
        await consume()
    assert closed.is_set()
    assert any(
        p.tool_call_id == "lookup-1"
        for m in caught.value.all_messages()
        for p in m.parts
        if p.part_kind == "tool-call"
    )


async def test_runtime_screen_image_is_visible_to_exactly_one_model_request(runtime, script):
    script.steps = [
        [calls(("look_at_screen", "{}", "screen-1"))],
        [calls(("lookup", '{"item":"sample"}', "lookup-1"))],
        ["Inspected."],
    ]

    async def lookup(item):
        return {"value": item}

    register_lookup(runtime, lookup)
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(images=["data:image/png;base64,c2NyZWVu"])
        )
    ]
    assert_terminal(events)

    def images(messages):
        return [
            item
            for message in messages
            for part in message.parts
            if isinstance(part, (UserPromptPart, ToolReturnPart))
            for item in (part.content if isinstance(part.content, list) else [part.content])
            if isinstance(item, BinaryContent)
        ]

    assert images(script.requests[0]) == []
    assert [item.data for item in images(script.requests[1])] == [b"screen"]
    assert images(script.requests[2]) == []
    path = await runtime.sessions.get_message_path("compat")
    screen_tool = path[-1]["segments"][0]["tools"][0]
    assert screen_tool["output"] == "[Inspected current screen]"


@pytest.mark.parametrize("validation_error", [False, True])
async def test_runtime_emits_one_error_and_result_for_failed_tool(
    runtime, script, validation_error
):
    args = '{"wrong":"sample"}' if validation_error else '{"item":"sample"}'
    script.steps = [[calls(("lookup", args, "lookup-1"))], ["Unable to look up."]]
    executed = []

    async def lookup(item):
        executed.append(item)
        return {"error": "Unavailable", "error_code": "NOT_FOUND"}

    register_lookup(runtime, lookup)
    events = [e async for e in runtime.streaming.stream_message(request())]
    assert_terminal(events)
    tool_events = [e for e in events if e["type"] in {"tool_call", "tool_error", "tool_result"}]
    assert [e["type"] for e in tool_events] == ["tool_call", "tool_error", "tool_result"]
    assert executed == ([] if validation_error else ["sample"])
