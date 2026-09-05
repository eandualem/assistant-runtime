"""History processing and persisted application state are different contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
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

from .test_execution import assert_terminal, calls, request


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
