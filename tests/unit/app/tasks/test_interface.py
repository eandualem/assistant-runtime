"""TaskService: background turns, their order and caps, cancellation and restarts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from assistant_runtime.app.assistant.config import AssistantConfig, TunableOverrides
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.exceptions import (
    AgentConflictError,
    AgentNotFoundError,
    TaskError,
    TaskLimitError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.app.tasks.models import AgentMessageRecord, TaskRecord
from assistant_runtime.base.events import EventHub
from assistant_runtime.principal import Principal
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError

ALICE = Principal(id="alice")
BOB = Principal(id="bob")
ADMIN = Principal(id="root", roles=frozenset({"admin"}))


class FakeStreaming:
    """Runs a turn by waiting on its task's gate; records what ran and when."""

    def __init__(self) -> None:
        self.gates: dict[str, asyncio.Event] = {}
        self.started: list[str] = []
        self.sessions: list[str] = []
        self.configs: list[dict] = []
        self.pins: dict[str, int] = {}
        self.running = 0
        self.peak = 0
        self.fail: dict[str, Exception] = {}
        self.pending: dict[str, dict] = {}

    def validate_profile(self, name):
        if name == "nope":
            raise UnknownProfileError("Unknown assistant profile 'nope'")

    def pin_session(self, session_id):
        self.pins[session_id] = self.pins.get(session_id, 0) + 1

    def unpin_session(self, session_id):
        self.pins[session_id] = self.pins.get(session_id, 0) - 1

    def gate(self, content: str) -> asyncio.Event:
        return self.gates.setdefault(content, asyncio.Event())

    async def run_message(self, request, *, principal=None):
        self.started.append(request.content)
        self.sessions.append(request.session_id)
        self.configs.append(
            request.config.model_dump(exclude_none=True) if request.config else None
        )
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await self.gate(request.content).wait()
            if request.content in self.fail:
                raise self.fail[request.content]
            return SimpleNamespace(
                content=f"done: {request.content}",
                usage={"requests": 1},
                pending_tool_call=self.pending.get(request.content),
            )
        finally:
            self.running -= 1


async def _service(**config) -> tuple[TaskService, FakeStreaming, list[dict]]:
    hub, seen = EventHub(), []
    hub.subscribe(seen.append)
    streaming = FakeStreaming()
    service = TaskService(TasksConfig(enabled=True, **config), streaming, events=hub)
    await service.start()
    return service, streaming, seen


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


class TestRunning:
    async def test_a_task_runs_in_the_background_and_announces_its_end(self):
        service, streaming, seen = await _service()
        record = await service.start_task("Summarise", parent_session_id="chat-1", principal=ALICE)
        assert (record.status, record.session_id) == ("queued", f"task-{record.id}")
        await _settle()
        assert (await service.get(record.id, ALICE)).status == "running"
        streaming.gate("Summarise").set()
        await _settle()
        done = await service.get(record.id, ALICE)
        assert (done.status, done.result, done.usage) == (
            "done",
            "done: Summarise",
            {"requests": 1},
        )
        assert seen[0]["type"] == "task_finished"
        assert (seen[0]["task_id"], seen[0]["status"], seen[0]["parent_session_id"]) == (
            record.id,
            "done",
            "chat-1",
        )

    async def test_context_is_appended_to_the_task(self):
        service, streaming, _ = await _service()
        await service.start_task("Review", context="Only the parser", principal=ALICE)
        await _settle()
        assert streaming.started == ["Review\n\nContext:\nOnly the parser"]

    async def test_tasks_about_one_subject_run_one_at_a_time_in_order(self):
        service, streaming, _ = await _service()
        await service.start_task("A1", subject="agent-a", principal=ALICE)
        await service.start_task("A2", subject="agent-a", principal=ALICE)
        await service.start_task("B1", subject="agent-b", principal=ALICE)
        await _settle()
        assert streaming.started == ["A1", "B1"]
        streaming.gate("A1").set()
        await _settle()
        assert streaming.started == ["A1", "B1", "A2"]

    async def test_at_most_max_concurrent_run_at_once(self):
        service, streaming, _ = await _service(max_concurrent=2)
        for name in ("one", "two", "three"):
            await service.start_task(name, principal=ALICE)
        await _settle()
        assert streaming.started == ["one", "two"]
        streaming.gate("one").set()
        await _settle()
        assert streaming.started == ["one", "two", "three"]
        assert streaming.peak == 2

    async def test_a_failing_turn_or_a_timeout_fails_the_task(self):
        service, streaming, seen = await _service(timeout_seconds=10)
        streaming.fail["boom"] = RuntimeError("model down")
        failing = await service.start_task("boom", principal=ALICE)
        streaming.gate("boom").set()
        await _settle()
        record = await service.get(failing.id, ALICE)
        assert (record.status, record.error) == ("failed", "model down")
        with patch("assistant_runtime.app.tasks.interface.asyncio.timeout") as timeout:
            timeout.side_effect = lambda _s: _RaisingTimeout()
            slow = await service.start_task("slow", principal=ALICE)
            await _settle()
        record = await service.get(slow.id, ALICE)
        assert record.status == "failed"
        assert "Timed out" in record.error

    async def test_a_turn_that_waits_on_the_host_fails_the_task(self):
        service, streaming, _ = await _service()
        streaming.pending["act"] = {"tool_name": "select_item", "call_id": "c1"}
        record = await service.start_task("act", principal=ALICE)
        streaming.gate("act").set()
        await _settle()
        failed = await service.get(record.id, ALICE)
        assert failed.status == "failed"
        assert "select_item" in failed.error

    async def test_the_default_profile_is_resolved_so_one_subject_stays_serial(self):
        hub = EventHub()
        streaming = FakeStreaming()
        service = TaskService(
            TasksConfig(enabled=True), streaming, events=hub, default_profile="neutral"
        )
        await service.start()
        implicit = await service.start_task("first", subject="agent-a", principal=ALICE)
        await service.start_task("second", profile="neutral", subject="agent-a", principal=ALICE)
        await _settle()
        assert implicit.profile == "neutral"
        assert streaming.started == ["first"]


class _RaisingTimeout:
    async def __aenter__(self):
        raise TimeoutError

    async def __aexit__(self, *exc):
        return False


class TestControl:
    async def test_cancel_stops_a_running_or_queued_task(self):
        service, _, seen = await _service(max_concurrent=1)
        running = await service.start_task("first", principal=ALICE)
        queued = await service.start_task("second", principal=ALICE)
        await _settle()
        assert (await service.cancel(queued.id, ALICE)).status == "cancelled"
        assert (await service.cancel(running.id, ALICE)).status == "cancelled"
        assert {e["status"] for e in seen} == {"cancelled"}

    async def test_cancelling_before_the_worker_starts_still_finishes_the_task(self):
        service, _, seen = await _service(max_waiting=1)
        record = await service.start_task("x", principal=ALICE)
        cancelled = await service.cancel(record.id, ALICE)  # no await in between: never ran
        assert cancelled.status == "cancelled"
        assert [e["status"] for e in seen] == ["cancelled"]
        await service.start_task("next", principal=ALICE)  # the waiting place was freed

    async def test_concurrent_starts_respect_the_waiting_limit(self):
        service, _, _ = await _service(max_waiting=1)
        store = service._store
        create = store.create

        async def slow_create(record):
            await asyncio.sleep(0.01)
            return await create(record)

        store.create = slow_create
        results = await asyncio.gather(
            *(service.start_task(f"t{n}", principal=ALICE) for n in range(5)),
            return_exceptions=True,
        )
        assert sum(not isinstance(r, Exception) for r in results) == 1
        assert sum(isinstance(r, TaskLimitError) for r in results) == 4

    async def test_stopping_the_service_interrupts_what_runs(self):
        service, _, seen = await _service()
        await service.start_task("long", principal=ALICE)
        await _settle()
        await service.stop()
        assert seen[0]["status"] == "interrupted"

    async def test_a_task_ended_elsewhere_keeps_that_end(self):
        """Another process that recovered the task first recorded its end; nothing flips back."""
        service, streaming, seen = await _service(max_concurrent=1)
        running = await service.start_task("first", principal=ALICE)
        waiting = await service.start_task("second", principal=ALICE)
        await _settle()
        for record in (running, waiting):
            await service._store.update(record.id, status="interrupted", error="recovered")
        streaming.gate("first").set()
        streaming.gate("second").set()
        await _settle()
        assert [(await service.get(r.id, ALICE)).status for r in (running, waiting)] == [
            "interrupted",
            "interrupted",
        ]
        assert seen == []  # the end was published where it was recorded, not again here
        assert streaming.started == ["first"]  # the waiting task never ran

    async def test_a_restart_marks_unfinished_tasks_interrupted(self):
        left = TaskRecord(
            id="t1", session_id="task-t1", task="x", status="interrupted", created_by="a"
        )

        class Store:
            durable = True

            async def mark_unfinished(self, status, error):
                assert status == "interrupted"
                return [left]

            async def mark_unfinished_messages(self, status, error):
                return []

            async def list_agents(self, **filters):
                return []

        hub, seen = EventHub(), []
        hub.subscribe(seen.append)
        service = TaskService(
            TasksConfig(enabled=True),
            FakeStreaming(),
            database_service=SimpleNamespace(healthy=True),
            events=hub,
        )
        with patch("assistant_runtime.app.tasks.interface.DatabaseTaskStore", return_value=Store()):
            await service.start()
        assert [(e["task_id"], e["status"]) for e in seen] == [("t1", "interrupted")]


class TestAdmission:
    async def test_disabled_tasks_are_refused(self):
        service = TaskService(TasksConfig(), FakeStreaming())
        await service.start()
        with pytest.raises(TasksDisabledError):
            await service.start_task("x")

    async def test_names_are_checked_before_anything_runs(self):
        service, _, _ = await _service()
        with pytest.raises(UnknownProfileError):
            await service.start_task("x", profile="nope")
        with pytest.raises(TaskError, match="Invalid profile or subject"):
            await service.start_task("x", subject="not a subject")
        with pytest.raises(TaskError, match="description"):
            await service.start_task("   ")

    async def test_too_many_waiting_tasks_are_refused(self):
        service, _, _ = await _service(max_waiting=2, max_concurrent=1)
        await service.start_task("one", principal=ALICE)
        await service.start_task("two", principal=ALICE)
        with pytest.raises(TaskLimitError):
            await service.start_task("three", principal=ALICE)

    async def test_callers_see_their_own_tasks_and_an_administrator_all(self):
        service, _, _ = await _service()
        mine = await service.start_task("mine", principal=ALICE)
        await service.start_task("theirs", principal=BOB)
        with pytest.raises(TaskNotFoundError):
            await service.get(mine.id, BOB)
        assert [r.task for r in await service.list(ALICE)] == ["mine"]
        assert {r.task for r in await service.list(ADMIN)} == {"mine", "theirs"}


class TestTools:
    async def test_the_tools_start_and_read_tasks_for_the_calling_session(self):
        from assistant_runtime.app.tasks._tools import register_task_tools
        from assistant_runtime.services.tools.request_context import assistant_request_context

        service, streaming, _ = await _service()
        handlers = {}
        tools = SimpleNamespace(
            register_backend_tool=lambda definition, handler: handlers.__setitem__(
                definition.name, handler
            )
        )
        register_task_tools(tools, service)
        assert set(handlers) == {
            "start_task",
            "list_tasks",
            "get_task",
            "cancel_task",
            "message_agent",
            "list_agents",
            "get_agent_message",
        }
        with assistant_request_context("chat-1", principal=ALICE):
            started = await handlers["start_task"](task="Look", subject="agent-a")
            listed = await handlers["list_tasks"]()
            refused = await handlers["start_task"](task="x", profile="nope")
        assert started["success"]
        assert started["status"] == "queued"
        assert [t["task_id"] for t in listed["tasks"]] == [started["task_id"]]
        assert refused["success"] is False
        streaming.gate("Look").set()
        await _settle()
        with assistant_request_context("chat-1", principal=ALICE):
            record = await handlers["get_task"](task_id=started["task_id"])
        assert (record["status"], record["result"], record["parent_session_id"]) == (
            "done",
            "done: Look",
            "chat-1",
        )


class TestAgents:
    async def test_messages_run_in_the_agents_one_session_in_order(self):
        service, streaming, seen = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        first = await service.message_agent(
            "m1", agent_id=agent.id, parent_session_id="main-1", principal=ALICE
        )
        await service.message_agent("m2", agent_id=agent.id, principal=ALICE)
        await _settle()
        assert streaming.started == ["m1"]
        streaming.gate("m1").set()
        await _settle()
        assert streaming.started == ["m1", "m2"]
        assert streaming.sessions == [agent.session_id, agent.session_id]
        done = await service.get_message(first.id, ALICE)
        assert (done.status, done.result, done.session_id) == (
            "done",
            "done: m1",
            agent.session_id,
        )
        assert seen[0]["type"] == "agent_message_finished"
        assert (seen[0]["message_id"], seen[0]["agent_id"], seen[0]["parent_session_id"]) == (
            first.id,
            agent.id,
            "main-1",
        )

    async def test_one_active_agent_per_identity_and_per_session(self):
        service, _, _ = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        with pytest.raises(AgentConflictError):
            await service.start_agent(subject="agent-a", principal=BOB)
        with pytest.raises(AgentConflictError):
            await service.start_agent(subject="agent-b", session_id=agent.session_id)
        await service.stop_agent(agent.id, ALICE)
        again = await service.start_agent(subject="agent-a", principal=ALICE)
        assert again.id != agent.id
        await service.start_agent(principal=ALICE)  # no profile and no subject is one identity too
        with pytest.raises(AgentConflictError):
            await service.start_agent(principal=ALICE)

    async def test_an_agents_session_stays_cached_until_it_moves_or_stops(self):
        service, streaming, _ = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        assert streaming.pins == {agent.session_id: 1}
        moved = await service.move_agent(agent.id, principal=ALICE)
        assert streaming.pins == {agent.session_id: 0, moved.session_id: 1}
        await service.stop_agent(agent.id, ALICE)
        assert streaming.pins == {agent.session_id: 0, moved.session_id: 0}

    async def test_a_move_waits_for_a_message_that_is_starting(self):
        """A message that read the agent's session starts there before a move returns."""
        service, _, _ = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        store = service._store
        update_message, release = store.update_message, asyncio.Event()

        async def slow_update_message(message_id, **values):
            if values.get("status") == "running":
                await release.wait()
            return await update_message(message_id, **values)

        store.update_message = slow_update_message
        record = await service.message_agent("m1", agent_id=agent.id, principal=ALICE)
        await _settle()
        move = asyncio.create_task(service.move_agent(agent.id, principal=ALICE))
        await _settle()
        assert not move.done()
        release.set()
        await move
        started = await service.get_message(record.id, ALICE)
        assert (started.status, started.session_id) == ("running", agent.session_id)

    async def test_a_move_applies_to_messages_not_yet_started(self):
        service, streaming, _ = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        await service.message_agent("m1", agent_id=agent.id, principal=ALICE)
        await service.message_agent("m2", agent_id=agent.id, principal=ALICE)
        await _settle()
        moved = await service.move_agent(agent.id, principal=ALICE)
        assert moved.session_id != agent.session_id
        streaming.gate("m1").set()
        streaming.gate("m2").set()
        await _settle()
        assert streaming.sessions == [agent.session_id, moved.session_id]
        named = await service.move_agent(agent.id, session_id="fresh-1", principal=ALICE)
        assert named.session_id == "fresh-1"

    async def test_a_config_change_applies_to_messages_not_yet_started(self):
        service, streaming, _ = await _service()
        agent = await service.start_agent(
            subject="agent-a",
            config=TunableOverrides(default_model="openai:gpt-5.1", codex_service_tier="fast"),
            principal=ALICE,
        )
        assert agent.to_dict()["config"] == {
            "default_model": "openai:gpt-5.1",
            "codex_service_tier": "fast",
        }
        await service.message_agent("m1", agent_id=agent.id, principal=ALICE)
        await service.message_agent("m2", agent_id=agent.id, principal=ALICE)
        await _settle()
        # Left out is unchanged, None clears, a value sets.
        configured = await service.configure_agent(
            agent.id,
            TunableOverrides.model_validate({"default_model": None, "temperature": 0.2}),
            principal=ALICE,
        )
        assert configured.config == {"codex_service_tier": "fast", "temperature": 0.2}
        streaming.gate("m1").set()
        streaming.gate("m2").set()
        await _settle()
        assert streaming.configs == [
            {"default_model": "openai:gpt-5.1", "codex_service_tier": "fast"},
            {"codex_service_tier": "fast", "temperature": 0.2},
        ]
        # An agent without a config still sends one, so no earlier config is reused.
        cleared = await service.configure_agent(
            agent.id,
            TunableOverrides.model_validate({"codex_service_tier": None, "temperature": None}),
            principal=ALICE,
        )
        assert cleared.config == {}
        await service.message_agent("m3", agent_id=agent.id, principal=ALICE)
        streaming.gate("m3").set()
        await _settle()
        assert streaming.configs[-1] == {}

    async def test_a_config_a_turn_would_ignore_is_refused(self):
        service = TaskService(
            TasksConfig(enabled=True),
            FakeStreaming(),
            assistant_config=AssistantConfig(
                request_models=["openai:gpt-5.1"], request_service_tier=False
            ),
        )
        await service.start()
        with pytest.raises(TaskError, match="default_model"):
            await service.start_agent(
                subject="agent-a", config=TunableOverrides(default_model="openai:other")
            )
        with pytest.raises(TaskError, match="codex_service_tier"):
            await service.start_agent(
                subject="agent-a", config=TunableOverrides(codex_service_tier="fast")
            )
        agent = await service.start_agent(
            subject="agent-a", config=TunableOverrides(subagent_model="openai:gpt-5.1")
        )
        with pytest.raises(TaskError, match="subagent_model"):
            await service.configure_agent(agent.id, TunableOverrides(subagent_model="openai:other"))
        assert (await service.get_agent(agent.id)).config == {"subagent_model": "openai:gpt-5.1"}
        await service.stop_agent(agent.id)
        with pytest.raises(AgentConflictError):
            await service.configure_agent(agent.id, TunableOverrides(temperature=0.2))

    async def test_stopping_cancels_its_messages_and_refuses_new_ones(self):
        service, streaming, seen = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        running = await service.message_agent("m1", agent_id=agent.id, principal=ALICE)
        queued = await service.message_agent("m2", agent_id=agent.id, principal=ALICE)
        await _settle()
        stopped = await service.stop_agent(agent.id, ALICE)
        assert stopped.status == "stopped"
        assert [(await service.get_message(m.id, ALICE)).status for m in (running, queued)] == [
            "cancelled",
            "cancelled",
        ]
        assert {e["status"] for e in seen} == {"cancelled"}
        assert streaming.started == ["m1"]
        with pytest.raises(AgentConflictError):
            await service.message_agent("m3", agent_id=agent.id, principal=ALICE)
        with pytest.raises(AgentConflictError):
            await service.move_agent(agent.id, principal=ALICE)

    async def test_messages_and_tasks_share_the_concurrency_limit(self):
        service, streaming, _ = await _service(max_concurrent=1)
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        await service.start_task("task", principal=ALICE)
        await service.message_agent("message", agent_id=agent.id, principal=ALICE)
        await _settle()
        assert streaming.started == ["task"]
        streaming.gate("task").set()
        await _settle()
        assert streaming.started == ["task", "message"]

    async def test_agents_are_found_by_profile_and_subject(self):
        hub = EventHub()
        service = TaskService(
            TasksConfig(enabled=True), FakeStreaming(), events=hub, default_profile="neutral"
        )
        await service.start()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        assert agent.profile == "neutral"
        record = await service.message_agent("hi", subject="agent-a", principal=ALICE)
        assert record.agent_id == agent.id
        with pytest.raises(AgentNotFoundError):
            await service.message_agent("hi", subject="agent-b", principal=ALICE)
        with pytest.raises(AgentNotFoundError):
            await service.message_agent("hi", subject="agent-a", principal=BOB)
        with pytest.raises(TaskError, match="content"):
            await service.message_agent("  ", agent_id=agent.id, principal=ALICE)

    async def test_callers_see_their_own_agents_and_an_administrator_all(self):
        service, _, _ = await _service()
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        with pytest.raises(AgentNotFoundError):
            await service.get_agent(agent.id, BOB)
        assert await service.list_agents(BOB) == []
        assert [a.id for a in await service.list_agents(ADMIN)] == [agent.id]

    async def test_disabled_agents_are_refused(self):
        service = TaskService(TasksConfig(), FakeStreaming())
        await service.start()
        with pytest.raises(TasksDisabledError):
            await service.start_agent(subject="agent-a")

    async def test_a_restart_marks_unfinished_messages_interrupted(self):
        left = AgentMessageRecord(
            id="m1", agent_id="a1", content="x", status="interrupted", created_by="a"
        )

        class Store:
            durable = True

            async def mark_unfinished(self, status, error):
                return []

            async def mark_unfinished_messages(self, status, error):
                assert status == "interrupted"
                return [left]

            async def list_agents(self, **filters):
                return []

        hub, seen = EventHub(), []
        hub.subscribe(seen.append)
        service = TaskService(
            TasksConfig(enabled=True),
            FakeStreaming(),
            database_service=SimpleNamespace(healthy=True),
            events=hub,
        )
        with patch("assistant_runtime.app.tasks.interface.DatabaseTaskStore", return_value=Store()):
            await service.start()
        assert [(e["type"], e["message_id"], e["status"]) for e in seen] == [
            ("agent_message_finished", "m1", "interrupted")
        ]

    async def test_the_tools_message_an_agent_for_the_calling_session(self):
        from assistant_runtime.app.tasks._tools import register_task_tools
        from assistant_runtime.services.tools.request_context import assistant_request_context

        service, streaming, _ = await _service()
        handlers = {}
        tools = SimpleNamespace(
            register_backend_tool=lambda definition, handler: handlers.__setitem__(
                definition.name, handler
            )
        )
        register_task_tools(tools, service)
        agent = await service.start_agent(subject="agent-a", principal=ALICE)
        with assistant_request_context("main-1", principal=ALICE):
            listed = await handlers["list_agents"]()
            sent = await handlers["message_agent"](message="Status?", subject="agent-a")
            missing = await handlers["message_agent"](message="Status?", subject="agent-z")
        assert listed["agents"] == [{"agent_id": agent.id, "profile": None, "subject": "agent-a"}]
        assert (sent["success"], sent["agent_id"], sent["status"]) == (True, agent.id, "queued")
        assert (missing["success"], missing["error_code"]) == (False, "agent_not_found")
        streaming.gate("Status?").set()
        await _settle()
        with assistant_request_context("main-1", principal=ALICE):
            record = await handlers["get_agent_message"](message_id=sent["message_id"])
        assert (record["status"], record["result"], record["parent_session_id"]) == (
            "done",
            "done: Status?",
            "main-1",
        )


async def test_agent_config_and_completed_usage_cannot_be_changed_through_returned_records():
    from assistant_runtime.app.assistant.config import TunableOverrides

    service, streaming, _ = await _service()
    try:
        agent = await service.start_agent(config=TunableOverrides(temperature=0.3))
        agent.config["temperature"] = 1.5
        reread = await service.get_agent(agent.id)
        assert reread.config == {"temperature": 0.3}
        reread.config.clear()
        configured = await service.configure_agent(agent.id, TunableOverrides(temperature=0.4))
        configured.config.clear()
        listed = await service.list_agents()
        listed[0].config.clear()
        assert (await service.get_agent(agent.id)).config == {"temperature": 0.4}
        task = await service.start_task("One task")
        streaming.gate("One task").set()
        await _settle()
        result = await service.get(task.id)
        assert result.status == "done"
        result.usage.clear()
        assert (await service.get(task.id)).usage == {"requests": 1}
    finally:
        await service.stop()


class TestReadPaging:
    async def _seeded(self):
        from datetime import UTC, datetime, timedelta

        service, _, _ = await _service()
        start = datetime(2026, 10, 1, tzinfo=UTC)
        for n in range(4):
            owner = "bob" if n == 2 else "alice"
            at = start + timedelta(minutes=n)
            await service._store.create(
                TaskRecord(
                    id=f"t{n}",
                    session_id=f"task-t{n}",
                    task=f"task {n}",
                    status="done",
                    created_by=owner,
                    created_at=at,
                )
            )
            await service._store.create_message(
                AgentMessageRecord(
                    id=f"m{n}",
                    agent_id="a2" if n == 2 else "a1",
                    content=f"message {n}",
                    status="done",
                    created_by=owner,
                    created_at=at,
                )
            )
        return service, start

    async def test_before_lists_the_next_older_tasks(self):
        service, _ = await self._seeded()
        assert [r.id for r in await service.list(ADMIN, before="t3", limit=2)] == ["t2", "t1"]
        assert [r.id for r in await service.list(ALICE, before="t3")] == ["t1", "t0"]
        assert [r.id for r in await service.list(ALICE, before="t0")] == []
        with pytest.raises(TaskNotFoundError):
            await service.list(ALICE, before="missing")
        with pytest.raises(TaskNotFoundError):  # another caller's task is no cursor
            await service.list(ALICE, before="t2")

    async def test_before_lists_the_next_older_agent_messages(self):
        service, _ = await self._seeded()
        older = await service.list_messages(ADMIN, agent_id="a1", before="m3")
        assert [r.id for r in older] == ["m1", "m0"]
        with pytest.raises(AgentNotFoundError):
            await service.list_messages(ADMIN, agent_id="a1", before="missing")
        with pytest.raises(AgentNotFoundError):  # a message of another agent
            await service.list_messages(ADMIN, agent_id="a1", before="m2")

    async def test_counts_cover_the_half_open_range(self):
        from datetime import timedelta

        service, start = await self._seeded()
        end = start + timedelta(minutes=3)
        assert await service.count(start, end) == 3
        assert await service.count_messages(start + timedelta(minutes=1), end) == 2
        await service.stop()
        assert await service.count(start, end) == 0  # not running: nothing to count
