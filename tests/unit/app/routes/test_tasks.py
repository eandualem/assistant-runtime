"""/tasks routes over a TaskService stand-in."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.tasks import router
from assistant_runtime.app.tasks.exceptions import (
    TaskLimitError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.models import TaskRecord

RECORD = TaskRecord(id="t1", session_id="task-t1", task="Look", status="queued", created_by="local")


def _app(**methods) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.task_service = SimpleNamespace(**methods)
    return app


async def _call(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.request(method, path, **kwargs)


async def test_start_returns_the_queued_record():
    async def start_task(task, **kwargs):
        assert (task, kwargs["subject"]) == ("Look", "agent-a")
        return RECORD

    response = await _call(
        _app(start_task=start_task), "POST", "/tasks", json={"task": "Look", "subject": "agent-a"}
    )
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


async def test_errors_map_to_statuses():
    async def disabled(*a, **k):
        raise TasksDisabledError("off")

    async def full(*a, **k):
        raise TaskLimitError("full")

    async def missing(*a, **k):
        raise TaskNotFoundError("no")

    assert (
        await _call(_app(start_task=disabled), "POST", "/tasks", json={"task": "x"})
    ).status_code == 503
    assert (
        await _call(_app(start_task=full), "POST", "/tasks", json={"task": "x"})
    ).status_code == 429
    assert (await _call(_app(get=missing), "GET", "/tasks/t9")).status_code == 404
    assert (await _call(_app(cancel=missing), "POST", "/tasks/t9/cancel")).status_code == 404


async def test_list_passes_the_filters():
    async def list_(principal, **kwargs):
        assert kwargs == {
            "parent_session_id": "chat-1",
            "status": "done",
            "limit": 5,
            "before": None,
            "updated_after": None,
        }
        return [RECORD]

    response = await _call(
        _app(list=list_),
        "GET",
        "/tasks",
        params={"parent_session_id": "chat-1", "status": "done", "limit": 5},
    )
    assert [r["id"] for r in response.json()] == ["t1"]


async def test_a_missing_service_is_503():
    app = FastAPI()
    app.include_router(router)
    assert (await _call(app, "GET", "/tasks")).status_code == 503


async def test_before_pages_and_an_unknown_cursor_is_404():
    async def list_(principal, **kwargs):
        if kwargs["before"] == "t9":
            raise TaskNotFoundError("No task 't9'")
        assert kwargs["before"] == "t2"
        return [RECORD]

    app = _app(list=list_)
    assert [r["id"] for r in (await _call(app, "GET", "/tasks?before=t2")).json()] == ["t1"]
    missing = await _call(app, "GET", "/tasks?before=t9")
    assert (missing.status_code, missing.json()["detail"]) == (404, "No task 't9'")


async def test_updated_after_is_a_cursor_and_excludes_before():
    seen = []

    async def list_(principal, **kwargs):
        seen.append(kwargs["updated_after"])
        return [RECORD]

    app = _app(list=list_)
    ok = await _call(app, "GET", "/tasks", params={"updated_after": "2026-10-01T12:00|t1"})
    both = await _call(app, "GET", "/tasks", params={"updated_after": "2026-10-01", "before": "t1"})
    bad = await _call(app, "GET", "/tasks", params={"updated_after": "soon"})
    assert ok.status_code == 200
    assert (seen[0].id, seen[0].updated_at.tzinfo is not None) == ("t1", True)
    assert (both.status_code, bad.status_code, len(seen)) == (422, 422, 1)
