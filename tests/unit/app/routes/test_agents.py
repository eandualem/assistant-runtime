"""/agents routes over a TaskService stand-in."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.access.deps import get_principal
from assistant_runtime.app.routes.agents import router
from assistant_runtime.app.tasks.exceptions import (
    AgentConflictError,
    AgentNotFoundError,
    TaskError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.models import AgentMessageRecord, AgentRecord
from assistant_runtime.principal import Principal

AGENT = AgentRecord(id="a1", session_id="agent-a1", status="active", created_by="local")
MESSAGE = AgentMessageRecord(
    id="m1", agent_id="a1", content="Status?", status="queued", created_by="local"
)


def _app(**methods) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.task_service = SimpleNamespace(**methods)
    return app


async def _call(app, method, path, **kwargs):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.request(method, path, **kwargs)


async def test_start_move_stop_and_message_an_agent():
    async def start_agent(**kwargs):
        assert (kwargs["subject"], kwargs["session_id"]) == ("agent-a", None)
        return AGENT

    async def move_agent(agent_id, **kwargs):
        assert (agent_id, kwargs["session_id"]) == ("a1", "fresh-1")
        return AGENT

    async def stop_agent(agent_id, principal):
        return AGENT

    async def message_agent(content, **kwargs):
        assert (content, kwargs["agent_id"], kwargs["parent_session_id"]) == (
            "Status?",
            "a1",
            "main-1",
        )
        return MESSAGE

    app = _app(
        start_agent=start_agent,
        move_agent=move_agent,
        stop_agent=stop_agent,
        message_agent=message_agent,
    )
    started = await _call(app, "POST", "/agents", json={"subject": "agent-a"})
    assert (started.status_code, started.json()["id"]) == (201, "a1")
    moved = await _call(app, "POST", "/agents/a1/session", json={"session_id": "fresh-1"})
    assert moved.status_code == 200
    assert (await _call(app, "POST", "/agents/a1/stop")).status_code == 200
    sent = await _call(
        app,
        "POST",
        "/agents/a1/messages",
        json={"content": "Status?", "parent_session_id": "main-1"},
    )
    assert (sent.status_code, sent.json()["status"]) == (202, "queued")


async def test_an_agents_config_is_set_at_start_and_patched():
    async def start_agent(**kwargs):
        assert kwargs["config"].model_dump(exclude_none=True) == {"default_model": "openai:m"}
        return AGENT

    async def configure_agent(agent_id, config, **kwargs):
        # Only what the body names reaches the service; null clears, the rest stay.
        assert (agent_id, config.model_dump(exclude_unset=True)) == (
            "a1",
            {"default_model": None, "codex_service_tier": "fast"},
        )
        return AGENT

    app = _app(start_agent=start_agent, configure_agent=configure_agent)
    started = await _call(app, "POST", "/agents", json={"config": {"default_model": "openai:m"}})
    assert (started.status_code, started.json()["config"]) == (201, {})
    patched = await _call(
        app,
        "PATCH",
        "/agents/a1",
        json={"config": {"default_model": None, "codex_service_tier": "fast"}},
    )
    assert patched.status_code == 200
    # A tunable outside ``config``, or an unknown one, is refused rather than ignored.
    for body in ({"default_model": "openai:m"}, {"config": {"model": "openai:m"}}):
        assert (await _call(app, "PATCH", "/agents/a1", json=body)).status_code == 422


async def test_administration_requires_the_admin_role():
    app = _app()
    app.dependency_overrides[get_principal] = lambda: Principal(id="bob")
    for path, body in (
        ("/agents", {}),
        ("/agents/a1/session", {}),
        ("/agents/a1/stop", None),
        ("/agents/a1/messages", {"content": "x"}),
    ):
        assert (await _call(app, "POST", path, json=body)).status_code == 403
    assert (await _call(app, "PATCH", "/agents/a1", json={})).status_code == 403


async def test_errors_map_to_statuses():
    async def disabled(*a, **k):
        raise TasksDisabledError("off")

    async def taken(*a, **k):
        raise AgentConflictError("taken")

    async def missing(*a, **k):
        raise AgentNotFoundError("no")

    async def refused(*a, **k):
        raise TaskError("Not allowed in a request's config")

    assert (await _call(_app(start_agent=disabled), "POST", "/agents", json={})).status_code == 503
    assert (await _call(_app(start_agent=taken), "POST", "/agents", json={})).status_code == 409
    assert (
        await _call(_app(configure_agent=refused), "PATCH", "/agents/a1", json={})
    ).status_code == 422
    assert (await _call(_app(get_agent=missing), "GET", "/agents/a9")).status_code == 404
    assert (
        await _call(_app(message_agent=taken), "POST", "/agents/a1/messages", json={"content": "x"})
    ).status_code == 409


async def test_messages_are_read_through_their_agent():
    async def get_agent(agent_id, principal):
        return AGENT

    async def list_messages(principal, **kwargs):
        assert kwargs == {
            "agent_id": "a1",
            "parent_session_id": None,
            "status": "done",
            "limit": 5,
            "before": None,
            "updated_after": None,
        }
        return [MESSAGE]

    async def get_message(message_id, principal):
        return MESSAGE

    app = _app(get_agent=get_agent, list_messages=list_messages, get_message=get_message)
    listed = await _call(app, "GET", "/agents/a1/messages", params={"status": "done", "limit": 5})
    assert [m["id"] for m in listed.json()] == ["m1"]
    assert (await _call(app, "GET", "/agents/a1/messages/m1")).json()["agent_id"] == "a1"
    assert (await _call(app, "GET", "/agents/a2/messages/m1")).status_code == 404


async def test_message_paging_passes_before_and_an_unknown_cursor_is_404():
    async def get_agent(agent_id, principal):
        return AGENT

    async def list_messages(principal, **kwargs):
        if kwargs["before"] == "m9":
            raise AgentNotFoundError("No agent message 'm9'")
        assert kwargs["before"] == "m2"
        return [MESSAGE]

    app = _app(get_agent=get_agent, list_messages=list_messages)
    listed = await _call(app, "GET", "/agents/a1/messages", params={"before": "m2"})
    assert [m["id"] for m in listed.json()] == ["m1"]
    assert (await _call(app, "GET", "/agents/a1/messages?before=m9")).status_code == 404


async def test_agent_messages_updated_after_excludes_before():
    app = _app(get_agent=AsyncMock(), list_messages=AsyncMock(return_value=[]))
    url = "/agents/a1/messages"
    ok = await _call(app, "GET", url, params={"updated_after": "2026-10-01T12:00Z|m1"})
    both = await _call(app, "GET", url, params={"updated_after": "2026-10-01", "before": "m1"})
    bad = await _call(app, "GET", url, params={"updated_after": "2026-10-01|"})
    assert (ok.status_code, ok.json()) == (200, [])
    assert (both.status_code, bad.status_code) == (422, 422)
