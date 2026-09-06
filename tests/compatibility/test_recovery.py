"""Persistence and recovery of sessions and pending host actions.

The database is the only boundary replaced: ``FakePersistence`` keeps rows in
memory the way Postgres would, so a "restart" is a cold session cache over the
same rows. Execution, planning, the runner and serialization are real.
"""

from __future__ import annotations

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
        self.messages[record["id"]] = copy.deepcopy(record)

    async def update_message(self, message_id, *, content=None, segments=None, usage=None):
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
