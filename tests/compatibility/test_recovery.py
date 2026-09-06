"""Persistence and recovery of sessions and pending host actions.

The database is the only boundary replaced: ``FakePersistence`` keeps rows in
memory the way Postgres would, so a "restart" is a cold session cache over the
same rows. Execution, planning, the runner and serialization are real.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic_ai.messages import ToolReturnPart

from assistant_runtime.app.assistant._session_persistence import LoadedSession
from assistant_runtime.app.assistant._stale_tools import STALE_HOST_TOOL_OUTPUT
from assistant_runtime.app.assistant.exceptions import SessionError

from .test_execution import assert_terminal, calls, request


@dataclass
class FakePersistence:
    """The ``SessionPersistence`` surface over in-memory rows."""

    sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    messages: dict[str, dict[str, Any]] = field(default_factory=dict)
    steering: dict[str, dict[str, Any]] = field(default_factory=dict)
    state_saves: list[dict[str, Any] | None] = field(default_factory=list)
    ops: list[tuple[str, Any]] = field(default_factory=list)
    """Every write, in order, for assertions about crash windows."""

    async def ensure_session(self, session_id, title, owner_id=None):
        self.sessions.setdefault(
            session_id,
            {
                "title": title,
                "owner_id": owner_id,
                "turn_number": 0,
                "working_memory": None,
                "pending_action": None,
            },
        )

    async def set_owner(self, session_id, owner_id):
        self.sessions[session_id]["owner_id"] = owner_id

    async def create_message(self, record):
        self.ops.append(("create_message", record["id"]))
        self.messages[record["id"]] = copy.deepcopy(record)

    async def update_message(self, message_id, *, content=None, segments=None, usage=None):
        self.ops.append(("update_message", message_id))
        row = self.messages[message_id]
        if content is not None:
            row["content"] = content
        if segments is not None:
            row["segments"] = copy.deepcopy(segments)
        if usage is not None:
            row["usage"] = usage

    async def update_segments(self, repaired):
        for message_id, segments in repaired:
            self.messages[message_id]["segments"] = copy.deepcopy(segments)

    async def create_steering(self, record):
        self.steering[record["id"]] = copy.deepcopy(record)

    async def mark_steering(self, steering_ids, *, status, delivered_at):
        for steering_id in steering_ids:
            self.steering[steering_id].update(status=status, delivered_at=delivered_at)

    async def save_state(self, session_id, ctx):
        from assistant_runtime.app.assistant._session_persistence import (
            pending_action_from_context,
        )

        row = self.sessions.setdefault(session_id, {"owner_id": ctx.get("owner_id")})
        row.update(
            title=ctx.get("title"),
            turn_number=ctx.get("turn_number", 0),
            working_memory=ctx.get("working_memory"),
            pending_action=pending_action_from_context(ctx),
        )
        self.state_saves.append(row["pending_action"])
        self.ops.append(("save_state", row["pending_action"]))

    async def session_for_telegram_chat(self, chat_id):
        return None

    async def delete(self, session_id):
        self.sessions.pop(session_id, None)

    async def list_sessions(self, limit, offset, *, owner_id=None):
        return []

    async def cleanup_expired(self):
        return 0

    async def load(self, session_id):
        row = self.sessions.get(session_id)
        if row is None:
            return None
        return LoadedSession(
            turn_number=row["turn_number"],
            working_memory=row["working_memory"],
            title=row["title"],
            owner_id=row["owner_id"],
            telegram_chat_id=None,
            telegram_bound_at=None,
            messages=[
                copy.deepcopy(m) for m in self.messages.values() if m["session_id"] == session_id
            ],
            steering=[
                copy.deepcopy(s) for s in self.steering.values() if s["session_id"] == session_id
            ],
            pending_action=copy.deepcopy(row["pending_action"]),
        )


@pytest.fixture
def persisted(runtime):
    """The runtime's session store writing through to fake rows."""
    fake = FakePersistence()
    runtime.sessions._db = fake
    return fake


def restart(runtime):
    """Forget every in-memory session, as a new process would."""
    runtime.sessions._sessions.clear()


def stored_tool(fake: FakePersistence, message_id: str, call_id: str) -> dict[str, Any]:
    return next(
        tool
        for segment in fake.messages[message_id]["segments"]
        if segment["kind"] == "tool_group"
        for tool in segment["tools"]
        if tool["id"] == call_id
    )


def continuation(**overrides):
    return request(
        **{
            "id": "continuation-1",
            "parent_id": "user-1",
            "content": "",
            "tool_call_id": "host-1",
            "tool_result": {"selected": True},
            **overrides,
        }
    )


async def start_pending_action(runtime, script, fake):
    script.steps = [[calls(("select_item", '{"item":"sample"}', "host-1"))], ["Selected."]]
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    assert first["pending_tool_call"]["call_id"] == "host-1"
    assert fake.sessions["compat"]["pending_action"] == {
        "tool_call_id": "host-1",
        "tool_name": "select_item",
        "assistant_message_id": first["message_id"],
        "batch": ["host-1"],
    }
    assert "output" not in stored_tool(fake, first["message_id"], "host-1")
    return first


async def test_pending_host_action_survives_restart_and_accepts_the_continuation(
    runtime, script, persisted
):
    first = await start_pending_action(runtime, script, persisted)

    restart(runtime)
    events = [e async for e in runtime.streaming.stream_message(continuation())]
    second = assert_terminal(events)

    assert second["message_id"] == first["message_id"]
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Selected."
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_call_id, p.outcome, p.content) for p in returns] == [
        ("host-1", "success", {"selected": True})
    ]
    stored = stored_tool(persisted, first["message_id"], "host-1")
    assert stored["output"] == {"selected": True}
    assert "status" not in stored
    assert persisted.sessions["compat"]["pending_action"] is None
    # A cold load after the continuation sees no pending action and nothing to repair.
    restart(runtime)
    ctx = await runtime.sessions.get_context_if_exists_async("compat")
    assert ctx["pending_tool_call_id"] is None
    assert stored_tool(persisted, first["message_id"], "host-1")["output"] == {"selected": True}


async def test_duplicate_continuation_cannot_reapply_a_recorded_result(runtime, script, persisted):
    first = await start_pending_action(runtime, script, persisted)
    assert_terminal([e async for e in runtime.streaming.stream_message(continuation())])
    requests_after_first = len(script.requests)

    for cold in (False, True):
        if cold:
            restart(runtime)
        events = [
            e
            async for e in runtime.streaming.stream_message(
                continuation(id=f"duplicate-{cold}", tool_result={"selected": False})
            )
        ]
        final = assert_terminal(events, error=True)
        assert final["error_type"] == "session_error"
        assert any("already recorded" in e.get("message", "") for e in events)
    assert len(script.requests) == requests_after_first
    assert stored_tool(persisted, first["message_id"], "host-1")["output"] == {"selected": True}


async def test_concurrent_duplicate_continuations_are_accepted_once(runtime, script, persisted):
    """Turns on a session are serialised, so the second duplicate plans after the first persisted."""
    first = await start_pending_action(runtime, script, persisted)

    async def collect(req):
        return [e async for e in runtime.streaming.stream_message(req)]

    results = await asyncio.gather(
        collect(continuation(id="c-1", tool_result={"selected": 1})),
        collect(continuation(id="c-2", tool_result={"selected": 2})),
    )
    finals = [assert_terminal(events, error=bool(i)) for i, events in enumerate(results)]
    assert finals[0]["message_id"] == first["message_id"]
    assert finals[1]["error_type"] == "session_error"
    assert any("already recorded" in e.get("message", "") for e in results[1])
    assert len(script.requests) == 2  # the host call, then the one accepted continuation
    assert stored_tool(persisted, first["message_id"], "host-1")["output"] == {"selected": 1}


async def test_duplicate_continuation_is_rejected_in_process_without_persistence(runtime, script):
    script.steps = [[calls(("select_item", '{"item":"sample"}', "host-1"))], ["Selected."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    assert_terminal([e async for e in runtime.streaming.stream_message(continuation())])
    with pytest.raises(SessionError, match="already recorded"):
        await runtime.streaming.run_message(continuation(id="duplicate"))


async def test_failed_host_action_is_recorded_as_failed(runtime, script, persisted):
    first = await start_pending_action(runtime, script, persisted)

    events = [
        e
        async for e in runtime.streaming.stream_message(
            continuation(tool_outcome="failed", tool_result={"error": "item not found"})
        )
    ]
    assert_terminal(events)
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert [(p.tool_call_id, p.outcome) for p in returns] == [("host-1", "failed")]
    assert "item not found" in str(returns[0].content)
    stored = stored_tool(persisted, first["message_id"], "host-1")
    assert stored["outcome"] == "failed"
    assert stored["status"] == "failed"
    assert persisted.sessions["compat"]["pending_action"] is None


async def test_reload_without_a_stored_pending_action_marks_the_call_unknown(
    runtime, script, persisted
):
    """Rows from before persisted pending actions, or a crash before the row was written."""
    first = await start_pending_action(runtime, script, persisted)
    persisted.sessions["compat"]["pending_action"] = None

    restart(runtime)
    ctx = await runtime.sessions.get_context_if_exists_async("compat")
    assert ctx["pending_tool_call_id"] is None
    stored = stored_tool(persisted, first["message_id"], "host-1")
    assert stored == {
        "id": "host-1",
        "name": "select_item",
        "input": {"item": "sample"},
        "output": STALE_HOST_TOOL_OUTPUT,
        "outcome": "interrupted",
        "status": "unknown",
    }
    events = [e async for e in runtime.streaming.stream_message(continuation())]
    final = assert_terminal(events, error=True)
    assert final["error_type"] == "session_error"
    assert any("already recorded (status: unknown)" in e.get("message", "") for e in events)


async def test_stored_pending_action_answered_before_the_row_was_cleared_is_dropped(
    runtime, script, persisted
):
    """A crash between recording the result and clearing the row must not reopen the action."""
    first = await start_pending_action(runtime, script, persisted)
    assert_terminal([e async for e in runtime.streaming.stream_message(continuation())])
    persisted.sessions["compat"]["pending_action"] = {
        "tool_call_id": "host-1",
        "tool_name": "select_item",
        "assistant_message_id": first["message_id"],
    }

    restart(runtime)
    ctx = await runtime.sessions.get_context_if_exists_async("compat")
    assert ctx["pending_tool_call_id"] is None
    assert persisted.sessions["compat"]["pending_action"] is None
    assert stored_tool(persisted, first["message_id"], "host-1")["output"] == {"selected": True}


async def test_new_message_before_the_continuation_records_the_action_as_superseded(
    runtime, script, persisted
):
    first = await start_pending_action(runtime, script, persisted)
    script.steps.append(["Moving on."])

    restart(runtime)
    assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                request(id="user-2", parent_id=first["message_id"], content="Never mind")
            )
        ]
    )
    stored = stored_tool(persisted, first["message_id"], "host-1")
    assert stored["outcome"] == "interrupted"
    assert stored["status"] == "superseded"
    assert persisted.sessions["compat"]["pending_action"] is None
    # Crash window: the action is resolved and the row cleared before the new
    # user message exists, so a restart in between cannot revive it next to
    # the message that superseded it.
    ops = persisted.ops
    assert ops.index(("update_message", first["message_id"])) < ops.index(
        ("create_message", "user-2")
    )
    assert ops.index(("save_state", None)) < ops.index(("create_message", "user-2"))
    events = [e async for e in runtime.streaming.stream_message(continuation())]
    assert_terminal(events, error=True)
    assert any("already recorded (status: superseded)" in e.get("message", "") for e in events)


async def test_repair_resolves_the_pending_action_as_unknown_on_the_row(runtime, script, persisted):
    first = await start_pending_action(runtime, script, persisted)

    report = await runtime.sessions.repair_stale_host_tools("compat")

    assert report["cleared_pending"] == {"tool_call_id": "host-1", "tool_name": "select_item"}
    assert report["repaired_tools"] == [first["message_id"]]
    assert persisted.sessions["compat"]["pending_action"] is None
    assert stored_tool(persisted, first["message_id"], "host-1")["status"] == "unknown"


async def test_parentless_message_during_a_live_turn_chains_after_it(runtime, script):
    """Ordinary messages are serialised per session: the replacement is planned only
    after the cancelled turn has persisted, so an omitted parent_id resolves to that
    turn's assistant row rather than producing a sibling of the first message."""
    script.steps = [["First answer."], ["Second answer."]]
    release = asyncio.Event()
    original_stream = script.stream

    async def holding_stream(messages, info):
        # The first turn streams its text, then stays live until released, so the
        # second request provably arrives while a turn is active.
        async for frame in original_stream(messages, info):
            yield frame
            if not release.is_set():
                await release.wait()

    script.stream = holding_stream
    first_events: list[dict] = []
    streamed = asyncio.Event()

    async def drain_first():
        async for event in runtime.streaming.stream_message(request(id="user-1", content="One")):
            first_events.append(event)
            if event["type"] == "text_delta":
                streamed.set()

    async def collect_second():
        return [
            e
            async for e in runtime.streaming.stream_message(
                request(id="user-2", content="Two", parent_id=None)
            )
        ]

    first = asyncio.create_task(drain_first())
    try:
        await asyncio.wait_for(streamed.wait(), 5)
        assert "compat" in runtime.streaming._active_turns  # the first turn is live
        second = asyncio.create_task(collect_second())
        await asyncio.sleep(0)
        release.set()
        second_events = await asyncio.wait_for(second, 5)
        await asyncio.wait_for(first, 5)
    finally:
        release.set()
        script.stream = original_stream

    assert second_events[-1]["status"] == "completed"
    first_final = next(e for e in first_events if e["type"] == "final_response")
    ctx = runtime.sessions.get_context("compat")
    assert ctx["children_by_parent"].get(None) == ["user-1"]  # one root
    assert ctx["message_index"]["user-2"]["parent_id"] == first_final["message_id"]
    assert [m["id"] for m in ctx["cached_path"]][:3] == [
        "user-1",
        first_final["message_id"],
        "user-2",
    ]


def two_host_calls():
    return calls(
        ("select_item", '{"item":"a"}', "host-1"), ("select_item", '{"item":"b"}', "host-2")
    )


async def test_two_host_calls_are_handed_over_one_at_a_time(runtime, script, persisted):
    script.steps = [[two_host_calls()], ["Both done."]]

    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    assert first["pending_tool_call"] == {
        "tool_name": "select_item",
        "call_id": "host-1",
        "arguments": {"item": "a"},
        "queued": ["host-2"],
    }
    assert persisted.sessions["compat"]["pending_action"]["batch"] == ["host-1", "host-2"]

    # The first result is recorded and the next call handed over without a model run.
    events = [
        e
        async for e in runtime.streaming.stream_message(
            continuation(id="c-1", tool_result={"selected": "a"})
        )
    ]
    handover = assert_terminal(events)
    assert handover["message_id"] == first["message_id"]
    assert handover["pending_tool_call"] == {
        "tool_name": "select_item",
        "call_id": "host-2",
        "arguments": {"item": "b"},
        "queued": [],
    }
    assert not any(e["type"] == "text_delta" for e in events)
    assert len(script.requests) == 1
    assert stored_tool(persisted, first["message_id"], "host-1")["output"] == {"selected": "a"}
    assert persisted.sessions["compat"]["pending_action"]["tool_call_id"] == "host-2"

    # A restart in between keeps the queue.
    restart(runtime)
    ctx = await runtime.sessions.get_context_if_exists_async("compat")
    assert ctx["pending_tool_call_id"] == "host-2"
    assert ctx["pending_tool_batch"] == ["host-1", "host-2"]
    assert "output" not in stored_tool(persisted, first["message_id"], "host-2")

    # The last result resumes the model with both results in the history.
    events = [
        e
        async for e in runtime.streaming.stream_message(
            continuation(id="c-2", tool_call_id="host-2", tool_result={"selected": "b"})
        )
    ]
    final = assert_terminal(events)
    assert final["message_id"] == first["message_id"]
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Both done."
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    # Both results reach the model; upstream decides their order in the resumed history.
    assert sorted((p.tool_call_id, p.content["selected"]) for p in returns) == [
        ("host-1", "a"),
        ("host-2", "b"),
    ]
    assert persisted.sessions["compat"]["pending_action"] is None
    assert len(await runtime.sessions.get_message_path("compat")) == 2


async def test_new_message_supersedes_every_queued_host_call(runtime, script, persisted):
    script.steps = [[two_host_calls()], ["Moving on."]]
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])

    assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                request(id="user-2", parent_id=first["message_id"], content="Never mind")
            )
        ]
    )

    for call_id in ("host-1", "host-2"):
        stored = stored_tool(persisted, first["message_id"], call_id)
        assert stored["outcome"] == "interrupted"
        assert stored["status"] == "superseded"
    assert persisted.sessions["compat"]["pending_action"] is None


async def test_failed_turn_resolves_unanswered_calls_so_the_next_message_works(
    runtime, script, persisted, monkeypatch
):
    """A turn that dies after the model issued tool calls must not block the next prompt."""
    from assistant_runtime.app.streaming import _runner

    original = _runner.pending_call_payloads

    def broken(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(_runner, "pending_call_payloads", broken)
    script.steps = [[calls(("select_item", '{"item":"a"}', "host-1"))], ["Recovered."]]

    events = [e async for e in runtime.streaming.stream_message(request())]
    assert_terminal(events, error=True)
    ctx = runtime.sessions.get_context("compat")
    (assistant_id,) = ctx["children_by_parent"]["user-1"]
    stored = stored_tool(persisted, assistant_id, "host-1")
    assert stored["outcome"] == "interrupted"
    assert stored["status"] == "cancelled"
    assert not ctx.get("pending_tool_call_id")

    monkeypatch.setattr(_runner, "pending_call_payloads", original)
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(id="user-2", parent_id=assistant_id, content="Again")
        )
    ]
    assert_terminal(events)
    assert "".join(e["content"] for e in events if e["type"] == "text_delta") == "Recovered."


async def test_a_long_batch_is_handed_over_in_model_order(runtime, script, persisted):
    ids = ["host-1", "host-2", "host-3", "host-4"]
    script.steps = [
        [calls(*(("select_item", f'{{"item":"{i}"}}', call_id) for i, call_id in enumerate(ids)))],
        ["All four."],
    ]
    first = assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    assert first["pending_tool_call"]["call_id"] == "host-1"
    assert first["pending_tool_call"]["queued"] == ["host-2", "host-3", "host-4"]

    handed = ["host-1"]
    for index, call_id in enumerate(ids[:-1]):
        final = assert_terminal(
            [
                e
                async for e in runtime.streaming.stream_message(
                    continuation(id=f"c-{index}", tool_call_id=call_id, tool_result={"n": index})
                )
            ]
        )
        handed.append(final["pending_tool_call"]["call_id"])
        assert final["pending_tool_call"]["queued"] == ids[index + 2 :]
        assert persisted.sessions["compat"]["pending_action"]["batch"] == ids
    assert handed == ids
    assert len(script.requests) == 1

    final = assert_terminal(
        [
            e
            async for e in runtime.streaming.stream_message(
                continuation(id="c-last", tool_call_id="host-4", tool_result={"n": 3})
            )
        ]
    )
    assert final.get("pending_tool_call") is None
    assert persisted.sessions["compat"]["pending_action"] is None
    assert len(script.requests) == 2
