"""Identity and scoped access through the real HTTP, in-process and turn paths."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.access import deps as access_deps
from assistant_runtime.app.access.config import AccessConfig
from assistant_runtime.app.access.deps import get_access_service as real_get_access_service
from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import AssistantDefinition, create_app, create_runtime
from assistant_runtime.principal import Credentials, Principal
from assistant_runtime.services.llm.interface import LlmService

from .test_execution import assert_terminal, request

ALICE = {"X-Assistant-Principal": "alice"}
BOB = {"X-Assistant-Principal": "bob"}
ROOT = {"X-Assistant-Principal": "root", "X-Assistant-Roles": "admin"}


@pytest.fixture
def real_access(monkeypatch):
    """Route tests default to the local operator; these use the configured mode."""
    monkeypatch.setattr(access_deps, "get_access_service", real_get_access_service)


@pytest.fixture
def scripted_model(monkeypatch, script):
    script.steps = [["One."], ["Two."], ["Three."], ["Four."]]
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    return script


def _message(session_id: str, message_id: str = "m1", parent_id: str | None = None) -> dict:
    return {"id": message_id, "session_id": session_id, "parent_id": parent_id, "content": "Hi"}


async def test_header_mode_isolates_principals_over_http(
    isolated_services, real_access, scripted_model
):
    settings = isolated_services.model_copy(update={"access": AccessConfig(mode="header")})
    app = create_app(settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        # Nobody without the proxy header.
        assert (await client.post("/api/chat", json=_message("s1"))).status_code == 401
        assert (await client.get("/api/sessions")).status_code == 401

        # Alice creates a session; it is hers.
        response = await client.post("/api/chat", json=_message("s1"), headers=ALICE)
        assert response.status_code == 200, response.text
        reply_id = response.json()["message_id"]
        session = (await client.get("/api/sessions/s1", headers=ALICE)).json()
        assert session["owner_id"] == "alice"

        # Bob can neither read, list, continue, cancel nor delete it.
        assert (await client.get("/api/sessions/s1", headers=BOB)).status_code == 403
        assert (await client.get("/api/sessions/s1/messages", headers=BOB)).status_code == 403
        assert (await client.get("/api/sessions/s1/tree", headers=BOB)).status_code == 403
        assert (await client.get("/api/sessions", headers=BOB)).json() == []
        follow_up = _message("s1", "m2", parent_id=reply_id)
        assert (await client.post("/api/chat", json=follow_up, headers=BOB)).status_code == 403
        assert (await client.post("/api/chat/s1/cancel", headers=BOB)).status_code == 403
        assert (await client.delete("/api/sessions/s1", headers=BOB)).status_code == 403
        # Nor can a client payload name another principal: the body is not identity.
        assert (
            await client.post("/api/chat", json={**follow_up, "principal": "alice"}, headers=BOB)
        ).status_code == 403

        # Alice continues her own conversation; an administrator sees everything.
        assert (await client.post("/api/chat", json=follow_up, headers=ALICE)).status_code == 200
        listed = (await client.get("/api/sessions", headers=ROOT)).json()
        assert [s["session_id"] for s in listed] == ["s1"]
        assert (await client.get("/api/sessions/s1/messages", headers=ROOT)).status_code == 200

        # Administration is separate from ordinary use.
        assert (await client.patch("/api/settings", json={}, headers=ALICE)).status_code == 403
        assert (await client.get("/api/settings", headers=ALICE)).status_code == 200
        assert (await client.get("/api/providers", headers=ALICE)).status_code == 403
        assert (await client.get("/api/providers", headers=ROOT)).status_code == 200
        assert (await client.get("/api/debug/tools", headers=ALICE)).status_code == 403
        assert (
            await client.patch("/api/artifacts/scratchpad", json={"content": "x"}, headers=ALICE)
        ).status_code == 403
        assert (await client.get("/api/artifacts/profile", headers=ALICE)).status_code == 200

        # Reassigning a session is administration; the new owner then has it.
        assert (
            await client.patch("/api/sessions/s1/owner", json={"owner_id": "bob"}, headers=ALICE)
        ).status_code == 403
        response = await client.patch(
            "/api/sessions/s1/owner", json={"owner_id": "bob"}, headers=ROOT
        )
        assert response.json() == {"session_id": "s1", "owner_id": "bob"}
        assert (await client.get("/api/sessions/s1", headers=BOB)).status_code == 200
        assert (await client.get("/api/sessions/s1", headers=ALICE)).status_code == 403


async def test_host_mode_uses_the_definition_callback(
    isolated_services, real_access, scripted_model
):
    async def by_token(credentials: Credentials) -> Principal | None:
        token = (credentials.header("authorization") or "").removeprefix("Bearer ").strip()
        return Principal(id=token) if token in {"alice", "bob"} else None

    settings = isolated_services.model_copy(update={"access": AccessConfig(mode="host")})
    app = create_app(assistant=AssistantDefinition(authenticate=by_token), settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app), base_url="http://test") as client,
    ):
        assert (await client.post("/api/chat", json=_message("s1"))).status_code == 401
        alice = {"Authorization": "Bearer alice"}
        assert (
            await client.post("/api/chat", json=_message("s1"), headers=alice)
        ).status_code == 200
        bob = {"Authorization": "Bearer bob"}
        assert (await client.get("/api/sessions/s1", headers=bob)).status_code == 403
        assert (await client.get("/api/sessions/s1", headers=alice)).status_code == 200


async def test_in_process_callers_are_the_local_operator_unless_they_say_otherwise(
    isolated_services, scripted_model
):
    async with create_runtime(settings=isolated_services) as runtime:
        first = await runtime.run_message(AssistantRequest(id="m1", session_id="s", content="Hi"))
        assert runtime._sessions.get_context("s")["owner_id"] == "local"
        # A named principal may not touch the local operator's session...
        with pytest.raises(AccessDeniedError):
            await runtime.run_message(
                AssistantRequest(id="m2", session_id="s", parent_id=first.message_id, content="x"),
                principal=Principal(id="bob"),
            )
        # ...but owns what it creates, and the unnamed caller (an admin) may read it.
        await runtime.run_message(
            AssistantRequest(id="m3", session_id="t", content="Hi"), principal=Principal(id="bob")
        )
        assert runtime._sessions.get_context("t")["owner_id"] == "bob"
        assert await runtime.cancel_session("t") is False
        with pytest.raises(AccessDeniedError):
            await runtime.cancel_session("t", principal=Principal(id="carol"))


async def test_streamed_turn_reports_forbidden_as_a_terminal_error(runtime, script):
    script.steps = [["One."]]
    assert_terminal([e async for e in runtime.streaming.stream_message(request())])
    events = [
        e
        async for e in runtime.streaming.stream_message(
            request(id="user-2", content="Again"), principal=Principal(id="mallory")
        )
    ]
    final = assert_terminal(events, error=True)
    assert final["error_type"] == "forbidden"
    error = next(e for e in events if e["type"] == "error")
    assert error["retry_allowed"] is False
    assert "belongs to another principal" in error["message"]
