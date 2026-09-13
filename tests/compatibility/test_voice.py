"""GPT-Live delegates into real Pydantic AI execution; only provider I/O is fake."""

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai.messages import ToolReturnPart

from assistant_runtime.app.access.deps import get_principal
from assistant_runtime.app.assistant.definition import AssistantDefinition
from assistant_runtime.app.voice.config import VoiceConfig
from assistant_runtime.main import create_app
from assistant_runtime.principal import Principal
from assistant_runtime.services.llm.interface import LlmService
from tests.unit.app.voice.test_voice import Transport, delegate, until

from .test_execution import calls

HOST_ACTION = {
    "name": "select_item",
    "description": "Select an item in the host.",
    "parameters": {
        "type": "object",
        "properties": {"item": {"type": "string"}},
        "required": ["item"],
    },
}


@pytest.fixture
async def voice_client(isolated_services, script, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-offline")
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: script.model())
    backend_calls = []

    def lookup_item() -> str:
        backend_calls.append("lookup")
        return "sample"

    settings = isolated_services.model_copy(
        update={"voice": VoiceConfig(enabled=True, close_timeout_seconds=0.05)}
    )
    app = create_app(settings=settings, assistant=AssistantDefinition(tools=[lookup_item]))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        transport = Transport()
        app.state.voice_service._transport = transport
        client.app = app
        client.transport = transport
        client.backend_calls = backend_calls
        yield client


async def start(client):
    response = await client.post(
        "/api/voice/calls",
        json={
            "session_id": "voice-thread",
            "sdp": "offer",
            "host_context": {"actions": [HOST_ACTION]},
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["call_id"]


async def test_backend_and_host_tools_share_native_pipeline_and_saved_history(voice_client, script):
    client = voice_client
    script.steps = [
        [calls(("lookup_item", "{}", "lookup-1"))],
        [calls(("select_item", '{"item":"sample"}', "host-1"))],
        ["Selected sample."],
    ]
    call_id = await start(client)
    connection = client.transport.connection
    delegate(connection, text="Find and select the sample")
    voice = client.app.state.voice_service
    await until(lambda: voice._calls[call_id].pending is not None)
    assert client.backend_calls == ["lookup"]
    sessions = client.app.state.assistant_service.get_session_store()
    assert sessions.get_context("voice-thread")["pending_tool_call_id"] == "host-1"
    response = await client.post(
        f"/api/voice/calls/{call_id}/delegations/item_1/tool-result",
        json={"tool_call_id": "host-1", "tool_result": {"selected": True}},
    )
    assert response.status_code == 202, response.text
    assert not sessions.get_context("voice-thread").get("pending_tool_call_id")
    assert client.backend_calls == ["lookup"]
    returns = [p for m in script.requests[-1] for p in m.parts if isinstance(p, ToolReturnPart)]
    assert any(p.tool_call_id == "host-1" and p.content == {"selected": True} for p in returns)
    assert connection.sent[-1]["content"] == "Selected sample."
    duplicate = await client.post(
        f"/api/voice/calls/{call_id}/delegations/item_1/tool-result",
        json={"tool_call_id": "host-1", "tool_result": {"selected": True}},
    )
    assert duplicate.status_code == 409
    await client.post(f"/api/voice/calls/{call_id}/close")
    history = (await client.get("/api/sessions/voice-thread/messages")).json()
    assert "Find and select" in history[0]["text"]
    assert history[-1]["text"] == "Selected sample."
    events = await client.get(f"/api/voice/calls/{call_id}/events")
    assert events.status_code == 200
    assert '"pending_host"' in events.text
    assert '"finalized": true' in events.text


async def test_voice_reservation_blocks_chat_repair_delete_and_reassignment(voice_client):
    client = voice_client
    call_id = await start(client)
    requests = [
        (
            "POST",
            "/api/chat",
            {"id": "intruder", "session_id": "voice-thread", "content": "replace"},
        ),
        ("POST", "/api/chat/voice-thread/cancel", None),
        ("POST", "/api/sessions/voice-thread/repair", None),
        ("DELETE", "/api/sessions/voice-thread", None),
        ("PATCH", "/api/sessions/voice-thread/owner", {"owner_id": "someone"}),
    ]
    for method, url, body in requests:
        response = await client.request(method, url, json=body)
        assert response.status_code == 409, (url, response.text)
    await client.post(f"/api/voice/calls/{call_id}/close")
    assert (await client.delete("/api/sessions/voice-thread")).status_code == 200
    assert (await client.get(f"/api/voice/calls/{call_id}")).status_code == 404
    await start(client)  # Reusing the session id must not resurrect deleted transcripts.
    assert (await client.get(f"/api/voice/calls/{call_id}")).status_code == 404


async def test_routes_enforce_owner_and_reject_provider_configuration(voice_client):
    client = voice_client
    for field in ("api_key", "model", "instructions", "principal", "delegation"):
        response = await client.post(
            "/api/voice/calls", json={"session_id": "x", "sdp": "offer", field: "untrusted"}
        )
        assert response.status_code == 422
    call_id = await start(client)
    client.app.dependency_overrides[get_principal] = lambda: Principal("someone-else")
    for method, suffix, body in [
        ("GET", "", None),
        ("GET", "/events", None),
        ("POST", "/close", None),
        ("POST", "/cancel", None),
        ("PATCH", "/context", {"host_context": {}}),
    ]:
        response = await client.request(method, f"/api/voice/calls/{call_id}{suffix}", json=body)
        assert response.status_code == 403
    client.app.dependency_overrides.clear()


async def test_cancel_pending_host_records_interrupted_not_failed(voice_client, script):
    client = voice_client
    script.steps = [[calls(("select_item", '{"item":"sample"}', "host-1"))]]
    call_id = await start(client)
    delegate(client.transport.connection)
    voice = client.app.state.voice_service
    await until(lambda: voice._calls[call_id].pending is not None)
    assert (await client.post(f"/api/voice/calls/{call_id}/cancel")).status_code == 200
    sessions = client.app.state.assistant_service.get_session_store()
    ctx = sessions.get_context("voice-thread")
    assert not ctx.get("pending_tool_call_id")
    history = await sessions.get_message_path("voice-thread")
    tools = [
        tool
        for m in history
        for s in (m["segments"] or [])
        if s["kind"] == "tool_group"
        for tool in s["tools"]
    ]
    assert tools[0]["outcome"] == "interrupted"
    assert tools[0]["status"] == "cancelled"
    response = await client.post(
        f"/api/voice/calls/{call_id}/delegations/item_1/tool-result",
        json={"tool_call_id": "host-1", "tool_result": "late"},
    )
    assert response.status_code == 409


async def test_cancel_admitted_continuation_preserves_result_without_starting_more_tools(
    voice_client, script, monkeypatch
):
    import asyncio

    from pydantic_ai.models.function import FunctionModel

    client = voice_client
    script.steps = [[calls(("select_item", '{"item":"sample"}', "host-1"))]]
    call_id = await start(client)
    delegate(client.transport.connection)
    voice = client.app.state.voice_service
    await until(lambda: voice._calls[call_id].pending is not None)
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def blocked_model(messages, info):
        started.set()
        try:
            await asyncio.Event().wait()
            yield calls(("lookup_item", "{}", "should-not-run"))
        finally:
            stopped.set()

    monkeypatch.setattr(script, "model", lambda: FunctionModel(stream_function=blocked_model))
    delivery = asyncio.create_task(
        client.post(
            f"/api/voice/calls/{call_id}/delegations/item_1/tool-result",
            json={"tool_call_id": "host-1", "tool_result": {"selected": True}},
        )
    )
    await asyncio.wait_for(started.wait(), 2)
    response = await asyncio.wait_for(client.post(f"/api/voice/calls/{call_id}/cancel"), 2)
    assert response.status_code == 200
    assert (await delivery).status_code == 202
    assert stopped.is_set()
    assert not client.backend_calls
    sessions = client.app.state.assistant_service.get_session_store()
    path = await sessions.get_message_path("voice-thread")
    tools = [
        t
        for m in path
        for s in (m["segments"] or [])
        if s["kind"] == "tool_group"
        for t in s["tools"]
    ]
    assert tools[0]["output"] == {"selected": True}
    assert tools[0].get("outcome", "success") == "success"


@pytest.mark.parametrize("mutation", ["delete", "owner", "repair"])
async def test_inflight_session_mutation_prevents_voice_reservation(
    voice_client, monkeypatch, mutation
):
    import asyncio

    client = voice_client
    call_id = await start(client)
    await client.post(f"/api/voice/calls/{call_id}/close")
    sessions = client.app.state.assistant_service.get_session_store()
    method, suffix, name, body = {
        "delete": ("DELETE", "", "delete_session", None),
        "owner": ("PATCH", "/owner", "set_owner", {"owner_id": "someone"}),
        "repair": ("POST", "/repair", "repair_stale_host_tools", None),
    }[mutation]
    entered, release = asyncio.Event(), asyncio.Event()
    original = getattr(sessions, name)

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(sessions, name, blocked)
    task = asyncio.create_task(
        client.request(method, f"/api/sessions/voice-thread{suffix}", json=body)
    )
    await asyncio.wait_for(entered.wait(), 2)
    response = await client.post(
        "/api/voice/calls", json={"session_id": "voice-thread", "sdp": "offer"}
    )
    assert response.status_code == 409
    assert len(client.transport.created) == 1
    release.set()
    assert (await task).status_code == 200
