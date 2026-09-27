"""TaskService: background turns, their order and caps, cancellation and restarts."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.exceptions import (
    TaskError,
    TaskLimitError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.app.tasks.models import TaskRecord
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
        self.running = 0
        self.peak = 0
        self.fail: dict[str, Exception] = {}
        self.pending: dict[str, dict] = {}

    def validate_profile(self, name):
        if name == "nope":
            raise UnknownProfileError("Unknown assistant profile 'nope'")

    def gate(self, content: str) -> asyncio.Event:
        return self.gates.setdefault(content, asyncio.Event())

    async def run_message(self, request, *, principal=None):
        self.started.append(request.content)
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
        assert set(handlers) == {"start_task", "list_tasks", "get_task", "cancel_task"}
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
