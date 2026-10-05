"""/counts over the in-memory stores: records created in a half-open time range."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.access.deps import get_principal
from assistant_runtime.app.assistant import AssistantRequest, SessionStore
from assistant_runtime.app.event_log.config import EventLogConfig
from assistant_runtime.app.event_log.interface import EventLogService
from assistant_runtime.app.routes.counts import router
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.app.tasks.models import AgentMessageRecord, TaskRecord
from assistant_runtime.principal import Principal
from assistant_runtime.services.actions.config import ActionsConfig
from assistant_runtime.services.actions.interface import ActionService

NOON = datetime(2026, 10, 1, 12, tzinfo=UTC)


async def _call(app: FastAPI, **params):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get("/counts", params=params)


async def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    events = EventLogService(EventLogConfig())
    actions = ActionService(ActionsConfig())
    tasks = TaskService(TasksConfig(), SimpleNamespace())
    for service in (events, actions, tasks):
        await service.start()
    await events.record(event_id="e1", direction="inbound", source="mail", kind="message")
    await actions.create(kind="message", by="local")
    await tasks._store.create(TaskRecord(**_task("t1")))
    await tasks._store.create(TaskRecord(**_task("t0", created_at=NOON)))
    await tasks._store.create_message(
        AgentMessageRecord(id="m1", agent_id="a1", content="x", status="done", created_by="local")
    )
    sessions = SessionStore()
    await sessions.register_user_message(AssistantRequest(id="u1", session_id="s1", content="Hi"))
    app.state.event_log_service = events
    app.state.action_service = actions
    app.state.task_service = tasks
    app.state.assistant_service = SimpleNamespace(get_session_store=lambda: sessions)
    return app


def _task(task_id: str, **values) -> dict:
    base = {"session_id": f"task-{task_id}", "task": "x", "status": "done", "created_by": "local"}
    return {"id": task_id, **base, **values}


async def test_counts_records_created_in_the_range():
    app = await _app()
    now = datetime.now(UTC)
    window = {
        "from": (now - timedelta(hours=1)).isoformat(),
        "to": (now + timedelta(hours=1)).isoformat(),
    }
    response = await _call(app, **window)
    earlier = await _call(app, **{"from": "2026-10-01T11:00", "to": "2026-10-01T12:00"})
    at_noon = await _call(app, **{"from": "2026-10-01T12:00", "to": "2026-10-01T13:00"})

    assert response.status_code == 200
    body = response.json()
    assert (body["from"], body["to"]) == (window["from"], window["to"])
    assert {k: v for k, v in body.items() if k not in ("from", "to")} == {
        "events": 1,
        "tasks": 1,
        "actions": 1,
        "agent_messages": 1,
        "messages": 1,
    }
    assert earlier.json()["tasks"] == 0  # the end is excluded
    assert at_noon.json()["tasks"] == 1  # the start is included; a naive value is UTC
    assert at_noon.json()["from"] == "2026-10-01T12:00:00+00:00"


async def test_absent_services_count_zero_and_the_range_is_checked():
    app = FastAPI()
    app.include_router(router)
    empty = await _call(app, **{"from": "2026-10-01T00:00Z", "to": "2026-10-02T00:00Z"})
    backwards = await _call(app, **{"from": "2026-10-02T00:00Z", "to": "2026-10-01T00:00Z"})
    missing = await _call(app, **{"from": "2026-10-01T00:00Z"})

    assert empty.json()["events"] == empty.json()["messages"] == 0
    assert (backwards.status_code, missing.status_code) == (422, 422)


async def test_counts_are_administration():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_principal] = lambda: Principal(id="bob")
    response = await _call(app, **{"from": "2026-10-01T00:00Z", "to": "2026-10-02T00:00Z"})
    assert response.status_code == 403
