"""Silent host decisions use real native output/deferred execution, offline."""

import json

import pytest
from pydantic_ai.models.function import DeltaThinkingPart, DeltaToolCall

from assistant_runtime.app.assistant.models import AssistantRequest


def request(request_id="decision", **kw):
    return AssistantRequest(
        id=request_id,
        session_id="body",
        content="Choose an action from the current state",
        output_mode="host_tools",
        host_context={
            "actions": [
                {
                    "name": "move",
                    "description": "Move the host",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        },
        **kw,
    )


def call(name="move", call_id="call"):
    return DeltaToolCall(
        name=name, json_args='{"decision":"hold"}' if name == "hold" else "{}", tool_call_id=call_id
    )


@pytest.mark.parametrize(
    "frames",
    [
        [{0: call("hold")}],
        [{0: call("hold", "hold"), 1: call("move", "move")}],
        [{0: call("move", "move"), 1: call("hold", "hold")}],
    ],
)
async def test_native_hold_wins_without_host_admission(runtime, script, frames):
    script.steps = [frames]
    result = await runtime.streaming.run_message(request())
    assert result.content is None
    assert result.decision == "hold"
    assert result.pending_tool_call is None
    assert not runtime.sessions.get_context("body").get("pending_tool_call_id")
    assert len(script.requests) == 1


@pytest.mark.parametrize("outcome", ["success", "failed"])
async def test_receipt_saves_without_model_and_inherits_mode(runtime, script, outcome):
    script.steps = [
        ["Forbidden preamble", {0: DeltaThinkingPart(content="Secret thought")}, {1: call()}]
    ]
    events = [e async for e in runtime.streaming.stream_message(request())]
    assert not any(
        e["type"].startswith("debug") or e["type"] in {"text_delta", "thinking_delta", "tool_call"}
        for e in events
    )
    final = next(e for e in events if e["type"] == "final_response")
    assert final["content"] is None
    assert final["decision"] == "pending"
    assert final["pending_tool_call"]["call_id"] == "call"
    ctx = runtime.sessions.get_context("body")
    assert ctx["pending_output_mode"] == "host_tools"
    assert "Forbidden preamble" not in json.dumps(ctx["message_index"], default=str)
    assert "Secret thought" not in json.dumps(ctx["message_index"], default=str)
    receipt = AssistantRequest(
        id="receipt",
        session_id="body",
        content="",
        tool_call_id="call",
        tool_result={"success": outcome == "success"},
        tool_outcome=outcome,
    )
    completed = await runtime.streaming.run_message(receipt)
    assert completed.decision == "completed"
    assert completed.content is None
    assert len(script.requests) == 1
    assert not ctx.get("pending_tool_call_id")
    assert "pending_output_mode" not in ctx
    with pytest.raises(Exception, match="already recorded"):
        await runtime.streaming.run_message(receipt.model_copy(update={"id": "duplicate"}))
    assert len(script.requests) == 1


async def test_multiple_actions_never_admitted(runtime, script):
    script.steps = [[{0: call(call_id="one"), 1: call(call_id="two")}]]
    events = [e async for e in runtime.streaming.stream_message(request())]
    assert not any(e.get("pending_tool_call") for e in events)
    assert any(e.get("error_type") == "invalid_decision" for e in events)
    assert not runtime.sessions.get_context("body").get("pending_tool_call_id")


async def test_only_request_actions_and_hold_are_offered(runtime, script):
    offered = []
    original = script.stream

    async def inspect(messages, info):
        offered.append(info)
        async for event in original(messages, info):
            yield event

    script.stream = inspect
    script.steps = [[{0: call("hold")}]]
    await runtime.streaming.run_message(request())
    assert [t.name for t in offered[0].function_tools] == ["move"]
    assert [t.name for t in offered[0].output_tools] == ["hold"]
    assert not offered[0].allow_text_output


async def test_cancellation_discards_partial_prose_and_never_admits_action(
    runtime, script, monkeypatch
):
    import asyncio

    from pydantic_ai.models.function import FunctionModel

    received = asyncio.Event()

    async def response(messages, info):
        yield "Forbidden partial text"
        received.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=response))
    events = []

    async def collect():
        async for event in runtime.streaming.stream_message(request()):
            events.append(event)

    task = asyncio.create_task(collect())
    try:
        async with asyncio.timeout(5):
            await received.wait()
            assert await runtime.streaming.cancel_session("body")
            await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert not any(e["type"] in {"text_delta", "thinking_delta", "tool_call"} for e in events)
    final = next(e for e in events if e["type"] == "final_response")
    assert final["error_type"] == "cancelled"
    assert not final.get("pending_tool_call")
    assert "Forbidden partial text" not in json.dumps(
        runtime.sessions.get_context("body")["message_index"], default=str
    )


async def test_no_actions_can_hold(runtime, script):
    script.steps = [[{0: call("hold")}]]
    result = await runtime.streaming.run_message(
        request().model_copy(update={"host_context": {"actions": []}})
    )
    assert result.decision == "hold"


async def test_receipt_failure_preserves_native_failed_outcome(runtime, script):
    from pydantic_ai.messages import ToolReturnPart

    script.steps = [[{0: call()}]]
    await runtime.streaming.run_message(request())
    await runtime.streaming.run_message(
        AssistantRequest(
            id="receipt",
            session_id="body",
            content="",
            tool_call_id="call",
            tool_result={"error": "host refused"},
            tool_outcome="failed",
        )
    )
    returns = [
        p
        for m in runtime.sessions.get_history("body")
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert any(p.tool_call_id == "call" and p.outcome == "failed" for p in returns)
    assert len(script.requests) == 1


async def test_http_decision_excludes_native_tools_and_auxiliary_work(
    isolated_services, monkeypatch
):
    from httpx import ASGITransport, AsyncClient
    from pydantic_ai import Tool
    from pydantic_ai.capabilities import ProcessHistory
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.toolsets import FunctionToolset

    from assistant_runtime.main import AssistantDefinition, create_app
    from assistant_runtime.services.history.interface import HistoryService
    from assistant_runtime.services.llm.interface import LlmService

    def forbidden(*args, **kwargs):
        raise AssertionError("Native or auxiliary work must not run in a host decision")

    def native_tool() -> str:
        return forbidden()

    calls = []

    async def response(messages, info):
        calls.append(messages)
        assert [t.name for t in info.function_tools] == ["move"]
        yield {0: call()}

    monkeypatch.setattr(
        LlmService, "_resolve_agent_model", lambda *_: FunctionModel(stream_function=response)
    )
    monkeypatch.setattr(HistoryService, "processor", forbidden)
    monkeypatch.setattr(HistoryService, "extract_memory_delta", forbidden)
    definition = AssistantDefinition(
        tools=[Tool(native_tool)],
        toolsets=[FunctionToolset([native_tool])],
        capabilities=[ProcessHistory(forbidden)],
        deps_factory=forbidden,
    )
    settings = isolated_services.model_copy(
        update={
            "assistant": isolated_services.assistant.model_copy(
                update={"enable_working_memory": True}
            )
        }
    )
    app = create_app(assistant=definition, settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        chosen = await client.post("/api/chat", json=request().model_dump())
        assert chosen.status_code == 200, chosen.text
        assert chosen.json()["decision"] == "pending"
        assert chosen.json()["content"] is None
        receipt = {
            "id": "receipt",
            "session_id": "body",
            "content": "",
            "tool_call_id": "call",
            "tool_result": {"success": True},
        }
        saved = await client.post("/api/chat", json=receipt)
        assert saved.status_code == 200, saved.text
        assert saved.json()["decision"] == "completed"
        assert saved.json()["content"] is None
        duplicate = await client.post("/api/chat", json={**receipt, "id": "duplicate"})
        assert duplicate.status_code == 409
    assert len(calls) == 1
