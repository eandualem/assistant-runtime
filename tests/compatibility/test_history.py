"""History processing and persisted application state are different contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset
from sqlalchemy.dialects.postgresql import dialect

from assistant_runtime.app.assistant._serialization import (
    assistant_record_to_flat_messages,
    build_assistant_message_content,
    path_records_to_model_history,
)
from assistant_runtime.app.assistant._session_persistence import LoadedSession
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant._stale_tools import STALE_HOST_TOOL_OUTPUT
from assistant_runtime.services.database.models import MessageORM
from assistant_runtime.services.history._manager import (
    COMPACTION_CACHE_KEY,
    SUMMARY_MARKER,
    TOOL_RESULT_PLACEHOLDER,
)
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.models import CompactionResult
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, calls, register_lookup, request

# 25k characters estimate above the smallest allowed budget of 5000 tokens.
LONG_TEXT = "x" * 25_000
SMALL_BUDGET = {"token_budget": 5000, "retain_recent": 1, "protect_recent_tool_results": 0}


def _texts(messages):
    return [
        getattr(part, "content", None)
        for message in messages
        for part in message.parts
        if isinstance(getattr(part, "content", None), str)
    ]


async def _first_turn_with_long_answer(runtime, script):
    script.steps = [[LONG_TEXT]]
    final = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    return {"id": "user-2", "parent_id": final["message_id"], "content": "Again"}


@pytest.mark.parametrize("args", [{"item": "sample"}, '{"item":"sample"}', '{"item":', "[]", None])
def test_serialization_uses_native_tool_argument_parsing(args):
    part = ToolCallPart("select_item", args, "host-1")
    _, segments, _ = build_assistant_message_content([ModelResponse(parts=[part])])
    expected = (
        {"item": "sample"}
        if isinstance(args, dict) or args == '{"item":"sample"}'
        else ({"INVALID_JSON": args} if args else {})
    )
    assert segments[0]["tools"][0]["input"] == expected


@pytest.mark.parametrize("outcome", ["success", "failed", "denied", "interrupted"])
@pytest.mark.parametrize("continuation", [False, True], ids=["normal-history", "continuation"])
def test_persisted_native_tool_outcomes_survive_both_history_paths(outcome, continuation):
    messages = [
        ModelResponse(parts=[ToolCallPart("lookup", "{}", "lookup-1")]),
        ModelRequest(parts=[ToolReturnPart("lookup", "Tool result", "lookup-1", outcome=outcome)]),
    ]
    content, segments, _ = build_assistant_message_content(messages)
    # Success stays implicit, preserving the schema of existing stored segments.
    assert ("outcome" in segments[0]["tools"][0]) is (outcome != "success")
    record = json.loads(json.dumps({"role": "assistant", "content": content, "segments": segments}))
    restored = (
        assistant_record_to_flat_messages(record)
        if continuation
        else path_records_to_model_history([record])
    )
    tool_return = next(
        part for message in restored for part in message.parts if isinstance(part, ToolReturnPart)
    )
    assert tool_return.outcome == outcome
    assert tool_return.content == "Tool result"
    assert tool_return.tool_call_id == "lookup-1"


class Product(BaseModel):
    code: str = Field(serialization_alias="sku")
    price: Decimal


@dataclass
class Delivery:
    product: Product
    due: date


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Product(code="lamp", price=Decimal("12.50")), {"sku": "lamp", "price": "12.50"}),
        (
            Delivery(Product(code="lamp", price=Decimal("12.50")), date(2026, 1, 2)),
            {"product": {"sku": "lamp", "price": "12.50"}, "due": "2026-01-02"},
        ),
        (
            {
                "product": Product(code="lamp", price=Decimal("12.50")),
                "images": [
                    (BinaryContent(data=b"\xff\xd8", media_type="image/jpeg"),),
                ],
            },
            {
                "product": {"sku": "lamp", "price": "12.50"},
                "images": [["[Binary content omitted]"]],
            },
        ),
    ],
    ids=["model", "dataclass", "nested-values-and-media"],
)
async def test_native_tool_outputs_are_json_compatible_at_database_boundary(
    script, value, expected
):
    original = copy.deepcopy(value)

    def lookup() -> Any:
        return value

    script.steps = [[calls(("lookup", "{}", "lookup-1"))], ["Ready."]]
    async with Agent(script.model(), tools=[lookup]).run_stream_events("Look up lamp") as stream:
        async for _ in stream:
            pass
    result = stream.result
    assert result.output == "Ready."
    _, segments, _ = build_assistant_message_content(result.new_messages())
    assert segments[0]["tools"][0]["output"] == expected
    # Exercise the actual JSONB encoding boundary without requiring Postgres.
    encode = MessageORM.__table__.c.segments.type.bind_processor(dialect())
    assert json.loads(encode(segments)) == segments
    assert value == original
    tool_return = next(
        part
        for message in script.requests[-1]
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    assert tool_return.content == original


@pytest.mark.parametrize("args", [{"item": "sample"}, '{"item":'])
async def test_public_history_closes_dangling_calls_including_malformed_args(script, args):
    history = [
        ModelRequest(parts=[UserPromptPart("Select an item")]),
        ModelResponse(parts=[ToolCallPart("select_item", args, "host-1")]),
        ModelRequest(parts=[UserPromptPart("Never mind")]),
        ModelResponse(parts=[TextPart("All right")]),
    ]
    original = copy.deepcopy(history)
    script.steps = [["Ready."]]
    async with Agent(script.model()).run_stream_events(
        "Continue", message_history=history
    ) as stream:
        events = [e async for e in stream]
    assert events[-1].result.output == "Ready."
    parts = [p for m in script.requests[0] for p in m.parts]
    call = next(p for p in parts if isinstance(p, ToolCallPart))
    returns = [p for p in parts if isinstance(p, ToolReturnPart)]
    assert len(returns) == 1
    assert returns[0].tool_call_id == call.tool_call_id == "host-1"
    assert returns[0].outcome == "interrupted"
    assert call.args == args
    assert history == original


async def test_public_history_drops_orphaned_results_before_model_request(script):
    """Observe the public run pipeline, not just its dangling-call helper."""
    history = [ModelRequest(parts=[ToolReturnPart("select_item", "unmatched", "orphan")])]
    script.steps = [["Ready."]]
    async with Agent(script.model()).run_stream_events(
        "Continue", message_history=history
    ) as stream:
        async for _ in stream:
            pass
    parts = [p for m in script.requests[0] for p in m.parts]
    assert not any(isinstance(p, ToolReturnPart) for p in parts)
    assert not any(isinstance(p, ToolCallPart) for p in parts)
    assert any(isinstance(p, UserPromptPart) and p.content == "Continue" for p in parts)
    assert history[0].parts[0].tool_call_id == "orphan"


async def test_public_process_history_can_filter_model_input_without_editing_source(script):
    history = [
        ModelRequest(parts=[UserPromptPart("Old context")]),
        ModelResponse(parts=[TextPart("Old answer")]),
    ]
    original = copy.deepcopy(history)
    processed = []

    def keep_current(messages):
        processed.append(copy.deepcopy(messages))
        return messages[-1:]

    script.steps = [["Current answer"]]
    agent = Agent(script.model(), capabilities=[ProcessHistory(keep_current)])
    async with agent.run_stream_events("Current request", message_history=history) as stream:
        async for _ in stream:
            pass
    assert len(processed) == 1
    assert len(processed[0]) == 3
    assert len(script.requests[0]) == 1
    assert script.requests[0][0].parts[0].content == "Current request"
    assert history == original


@pytest.mark.parametrize("pending_host", [False, True], ids=["completed-branch", "pending-host"])
async def test_session_reload_restores_tree_and_repairs_unfinished_host_action(
    runtime, script, pending_host
):
    script.steps = (
        [[calls(("select_item", '{"item":"sample"}', "host-1"))]]
        if pending_host
        else [["First answer"], ["Branch answer"]]
    )
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    if not pending_host:
        assert_terminal(
            [
                e
                async for e in runtime.streaming.stream_message(
                    request(id="user-2", parent_id="user-1", content="A different branch")
                )
            ]
        )
    ctx = runtime.sessions.get_context("compat")
    await runtime.sessions.queue_steering(
        "compat", request(id="steering-1", message_type="steering", content="Remember this")
    )
    snapshot = LoadedSession(
        turn_number=ctx["turn_number"],
        working_memory=ctx["working_memory"],
        title=ctx["title"],
        telegram_chat_id=None,
        telegram_bound_at=None,
        messages=copy.deepcopy(list(ctx["message_index"].values())),
        steering=copy.deepcopy(list(ctx["steering_index"].values())),
    )
    # Replace the persistence boundary only; exercise fresh-store hydration,
    # path reconstruction, stale repair and model-history conversion together.
    restored = SessionStore()
    restored._db = AsyncMock()
    restored._db.load.return_value = snapshot
    loaded = await restored.get_context_if_exists_async("compat")
    assert loaded["turn_number"] == ctx["turn_number"]
    assert loaded["active_leaf_id"] == ctx["active_leaf_id"]
    assert loaded["pending_steering_ids"] == ["steering-1"]
    assert loaded["pending_tool_call_id"] is None
    old_branch = await restored.get_message_path("compat", leaf_id=first["message_id"])
    assert [m["id"] for m in old_branch] == ["user-1", first["message_id"]]
    if pending_host:
        restored._db.update_segments.assert_awaited_once()
        parts = [p for m in restored.get_history("compat") for p in m.parts]
        call = next(p for p in parts if isinstance(p, ToolCallPart))
        result = next(p for p in parts if isinstance(p, ToolReturnPart))
        assert call.args_as_dict() == {"item": "sample"}
        assert call.tool_call_id == result.tool_call_id == "host-1"
        assert result.content == STALE_HOST_TOOL_OUTPUT
    else:
        restored._db.update_segments.assert_not_awaited()
        assert loaded["message_count"] == 4
        assert [m["id"] for m in loaded["cached_path"]][:-1] == ["user-1", "user-2"]


async def test_public_continuation_closes_older_dangling_call_with_deferred_result(
    script, host_schema
):
    """The runtime no longer repairs history itself; core handles both calls."""
    history = [
        ModelRequest(parts=[UserPromptPart("first")]),
        ModelResponse(parts=[ToolCallPart("select_item", '{"item":"a"}', "old-1")]),
        ModelRequest(parts=[UserPromptPart("second")]),
        ModelResponse(parts=[ToolCallPart("select_item", '{"item":"b"}', "host-1")]),
    ]
    original = copy.deepcopy(history)
    toolset = ExternalToolset(
        [
            ToolDefinition(name=name, parameters_json_schema=schema["parameters"])
            for name, schema in host_schema.items()
        ]
    )
    script.steps = [["Done."]]
    agent = Agent(script.model(), toolsets=[toolset], output_type=[str, DeferredToolRequests])
    async with agent.run_stream_events(
        None,
        message_history=history,
        deferred_tool_results=DeferredToolResults(calls={"host-1": {"ok": True}}),
    ) as stream:
        async for _ in stream:
            pass
    returns = {
        p.tool_call_id: p
        for m in script.requests[0]
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert returns["old-1"].outcome == "interrupted"
    assert returns["host-1"].outcome == "success"
    assert returns["host-1"].content == {"ok": True}
    assert history == original


@pytest.mark.parametrize("history_config", [HistoryConfig(**SMALL_BUDGET)], indirect=True)
async def test_runtime_clears_old_tool_results_in_model_input_only(runtime, script):
    async def lookup(item):
        return {"value": LONG_TEXT if item == "a" else "small"}

    register_lookup(runtime, lookup)
    script.steps = [[calls(("lookup", '{"item":"a"}', "lookup-1"))], ["First."]]
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    script.steps.extend([[calls(("lookup", '{"item":"b"}', "lookup-2"))], ["Second."]])
    follow_up = request(id="user-2", parent_id=first["message_id"], content="Again")
    assert_terminal([e async for e in runtime.streaming.stream_message(follow_up)])
    # Turn two: the first request already exceeds the budget through lookup-1.
    returns = {
        p.tool_call_id: p
        for m in script.requests[2]
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert returns["lookup-1"].content == TOOL_RESULT_PLACEHOLDER
    assert returns["lookup-1"].tool_name == "lookup"
    # The turn's own result is never cleared: the run must keep reporting it.
    returns = {
        p.tool_call_id: p
        for m in script.requests[3]
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    }
    assert {call: r.content for call, r in returns.items()} == {
        "lookup-1": TOOL_RESULT_PLACEHOLDER,
        "lookup-2": {"value": "small"},
    }
    assert SUMMARY_MARKER not in "".join(_texts(script.requests[3]))
    # Stored segments keep the real outputs.
    path = await runtime.sessions.get_message_path("compat")
    outputs = [
        tool["output"]["value"]
        for record in path
        for segment in record.get("segments") or []
        if segment["kind"] == "tool_group"
        for tool in segment["tools"]
    ]
    assert outputs == [LONG_TEXT, "small"]


@pytest.mark.parametrize("history_config", [HistoryConfig(**SMALL_BUDGET)], indirect=True)
async def test_runtime_summarizes_once_per_turn_through_native_execution(runtime, script):
    follow_up = await _first_turn_with_long_answer(runtime, script)

    executed = []

    async def lookup(item):
        executed.append(item)
        return {"value": "small"}

    register_lookup(runtime, lookup)
    script.steps.extend(
        [
            # The summarizer's own native run returns the structured summary.
            [calls(("final_result", '{"summary":"Earlier: a long answer"}', "sum-1"))],
            [calls(("lookup", '{"item":"sample"}', "lookup-1"))],
            ["Done."],
        ]
    )
    events = [e async for e in runtime.streaming.stream_message(request(**follow_up))]
    assert_terminal(events)
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Done."
    assert executed == ["sample"]
    assert len(script.requests) == 4
    summary_texts = [t for t in _texts(script.requests[1]) if "Earlier" not in t]
    assert any("Analyze the following conversation" in t for t in summary_texts)
    for model_input in script.requests[2:]:
        texts = _texts(model_input)
        assert texts[0] == "Help"
        assert texts[1].startswith(SUMMARY_MARKER)
        assert "Earlier: a long answer" in texts[1]
        assert LONG_TEXT not in texts
    assert _texts(script.requests[2])[2:] == ["Again"]
    tool_return = next(
        p for m in script.requests[3] for p in m.parts if isinstance(p, ToolReturnPart)
    )
    assert tool_return.content == {"value": "small"}
    ctx = runtime.sessions.get_context("compat")
    assert ctx[COMPACTION_CACHE_KEY]["prefix_count"] == 2
    assert not ctx.get("working_memory")
    path = await runtime.sessions.get_message_path("compat")
    assert path[1]["content"] == LONG_TEXT
    assert [m["role"] for m in path] == ["user", "assistant", "user", "assistant"]


@pytest.mark.parametrize("history_config", [HistoryConfig(**SMALL_BUDGET)], indirect=True)
async def test_runtime_summary_failure_falls_back_without_failing_the_turn(
    runtime, script, monkeypatch
):
    follow_up = await _first_turn_with_long_answer(runtime, script)
    build_agent = LlmService.build_agent

    def failing_summarizer(self, *args, **kwargs):
        agent = build_agent(self, *args, **kwargs)
        if kwargs.get("output_type") is CompactionResult:
            agent = MagicMock(run=AsyncMock(side_effect=RuntimeError("summarizer down")))
        return agent

    monkeypatch.setattr(LlmService, "build_agent", failing_summarizer)
    script.steps.append(["Done."])
    events = [e async for e in runtime.streaming.stream_message(request(**follow_up))]
    assert_terminal(events)
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Done."
    texts = _texts(script.requests[1])
    assert texts[1].startswith(SUMMARY_MARKER)
    assert "summarization failed" in texts[1]
    assert LONG_TEXT not in texts


@pytest.mark.parametrize(
    "history_config",
    [HistoryConfig(compaction_enabled=False, **SMALL_BUDGET)],
    indirect=True,
)
async def test_runtime_disabled_compaction_leaves_history_to_host_capabilities(runtime, script):
    follow_up = await _first_turn_with_long_answer(runtime, script)
    script.steps.append(["Done."])
    assert_terminal([e async for e in runtime.streaming.stream_message(request(**follow_up))])
    assert LONG_TEXT in _texts(script.requests[1])
    assert COMPACTION_CACHE_KEY not in runtime.sessions.get_context("compat")
