"""Codex voice provider: protocol mapping, guards and drift checks, all offline."""

import asyncio
import json
import os
import stat
import sys

import pytest

from assistant_runtime.app.voice._codex import (
    CodexTransport,
    _AppServer,
    missing_from_schema,
    usage_report,
)
from assistant_runtime.app.voice.config import VoiceConfig
from assistant_runtime.app.voice.exceptions import VoiceError
from assistant_runtime.app.voice.interface import VoiceService
from assistant_runtime.app.voice.models import VoiceOffer
from tests.unit.app.voice.test_voice import OWNER, Backend
from tests.voice_helpers import until

LIMITS = {
    "ordinaryUsageAllowed": True,
    "rateLimitsByLimitId": {
        "codex": {
            "credits": {"hasCredits": False, "unlimited": False, "balance": "0"},
            "spendControlReached": False,
            "primary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 1790000000},
        }
    },
}


def _limits(**changes):
    snapshot = {**LIMITS["rateLimitsByLimitId"]["codex"], **changes.pop("snapshot", {})}
    return {**LIMITS, **changes, "rateLimitsByLimitId": {"codex": snapshot}}


def test_usage_report_admits_included_usage_and_names_each_refusal():
    report = usage_report(LIMITS, 97)
    assert report["allowed"] is True
    assert report["reason"] is None
    assert report["windows"] == [
        {
            "limit": "codex",
            "window": "primary",
            "used_percent": 40,
            "window_minutes": 10080,
            "resets_at": 1790000000,
        }
    ]
    cases = {
        "credits_available": _limits(snapshot={"credits": {"hasCredits": True}}),
        "spend_control_reached": _limits(snapshot={"spendControlReached": True}),
        "usage_not_allowed": _limits(ordinaryUsageAllowed=False),
        "usage_window_limit": _limits(snapshot={"primary": {"usedPercent": 97}}),
        "usage_limit_reached": _limits(snapshot={"rateLimitReachedType": "rate_limit_reached"}),
    }
    for reason, limits in cases.items():
        report = usage_report(limits, 97)
        assert (report["allowed"], report["reason"]) == (False, reason)


def _schema(directory, *, drop_method=None, drop_param=None, versions=("v1", "v2", "v3")):
    def variants(methods):
        return {"oneOf": [{"properties": {"method": {"enum": [m]}}} for m in methods]}

    requests = [
        "account/read",
        "account/rateLimits/read",
        "thread/start",
        "thread/realtime/start",
        "thread/realtime/stop",
        "thread/realtime/appendSpeech",
        "thread/realtime/appendText",
        "turn/interrupt",
    ]
    notifications = [
        "thread/realtime/sdp",
        "thread/realtime/started",
        "thread/realtime/closed",
        "thread/realtime/error",
        "thread/realtime/transcript/delta",
        "thread/realtime/transcript/done",
        "thread/realtime/itemAdded",
        "turn/started",
    ]
    params = [
        "clientManagedHandoffs",
        "includeStartupContext",
        "initialItems",
        "outputModality",
        "prompt",
        "transport",
        "version",
        "voice",
    ]
    (directory / "v2").mkdir()
    (directory / "ClientRequest.json").write_text(
        json.dumps(variants([m for m in requests if m != drop_method]))
    )
    (directory / "ServerNotification.json").write_text(
        json.dumps(variants([m for m in notifications if m != drop_method]))
    )
    (directory / "v2" / "ThreadRealtimeStartParams.json").write_text(
        json.dumps(
            {
                "properties": {p: {} for p in params if p != drop_param},
                "definitions": {"RealtimeConversationVersion": {"enum": list(versions)}},
            }
        )
    )


def test_schema_check_names_what_an_installed_cli_lacks(tmp_path):
    for name, kwargs, missing in (
        ("complete", {}, []),
        (
            "no_speech",
            {"drop_method": "thread/realtime/appendSpeech"},
            ["thread/realtime/appendSpeech"],
        ),
        (
            "old_start",
            {"drop_param": "clientManagedHandoffs"},
            ["thread/realtime/start.clientManagedHandoffs"],
        ),
        ("no_v3", {"versions": ("v1", "v2")}, ["thread/realtime/start.version=v3"]),
    ):
        directory = tmp_path / name
        directory.mkdir()
        _schema(directory, **kwargs)
        assert missing_from_schema(directory) == missing
    assert "thread/start" in missing_from_schema(tmp_path / "absent")


class FakeServer:
    """Scripted app-server: records requests and feeds notifications per thread."""

    def __init__(self):
        self.requests = []
        self.queues = {}
        self.results = {
            "account/rateLimits/read": LIMITS,
            "thread/start": {"thread": {"id": "thread-1"}},
        }
        self.failures = set()

    async def ensure_started(self):
        return None

    async def stop(self):
        return None

    async def request(self, method, params):
        self.requests.append((method, params))
        if method in self.failures:
            raise VoiceError(f"Codex {method} failed", 502)
        if method == "thread/realtime/start":
            self.feed(params["threadId"], "thread/realtime/sdp", sdp="answer-sdp")
        if method == "thread/realtime/stop":
            self.feed(params["threadId"], "thread/realtime/closed", reason="requested")
        return self.results.get(method, {})

    def subscribe(self, thread_id):
        return self.queues.setdefault(thread_id, asyncio.Queue())

    def unsubscribe(self, thread_id):
        self.queues.pop(thread_id, None)

    def feed(self, thread_id, method, **params):
        self.subscribe(thread_id).put_nowait(
            {"method": method, "params": {"threadId": thread_id, **params}}
        )

    def sent(self, method):
        return [params for name, params in self.requests if name == method]


@pytest.fixture
def codex():
    transport = CodexTransport("codex", 1, usage_ceiling_percent=97, usage_check_seconds=60)
    transport._server = FakeServer()
    transport._compatibility = {"version": "0.157.1", "compatible": True, "missing": []}
    return transport


SESSION = {
    "instructions": "Be brief.",
    "voice": "cove",
    "history": [{"role": "user", "text": "Earlier"}],
}


async def test_create_starts_a_v3_webrtc_session_without_an_api_key(codex):
    thread, answer = await codex.create(None, SESSION, "offer-sdp")
    assert (thread, answer) == ("thread-1", "answer-sdp")
    [start] = codex._server.sent("thread/start")
    assert start["ephemeral"] is True
    assert start["sandbox"] == "read-only"
    [realtime] = codex._server.sent("thread/realtime/start")
    assert realtime == {
        "threadId": "thread-1",
        "outputModality": "audio",
        "transport": {"type": "webrtc", "sdp": "offer-sdp"},
        "prompt": "Be brief.",
        "version": "v3",
        "includeStartupContext": False,
        "clientManagedHandoffs": True,
        "voice": "cove",
        "initialItems": [{"role": "user", "text": "Earlier"}],
    }


async def test_create_refuses_before_allocation_with_a_machine_readable_reason(codex):
    codex._server.results["account/rateLimits/read"] = _limits(
        snapshot={"primary": {"usedPercent": 99}}
    )
    with pytest.raises(VoiceError) as refused:
        await codex.create(None, SESSION, "offer-sdp")
    assert refused.value.status_code == 409
    assert refused.value.metadata == {
        "allocation_status": "rejected",
        "reason": "usage_window_limit",
    }
    assert not codex._server.sent("thread/start")

    codex._compatibility = {"version": "0.100.0", "compatible": False, "missing": ["x"]}
    with pytest.raises(VoiceError) as incompatible:
        await codex.create(None, SESSION, "offer-sdp")
    assert incompatible.value.status_code == 503
    assert incompatible.value.metadata["reason"] == "codex_incompatible"


async def test_a_provider_refusal_stops_the_session_and_keeps_its_text_private(codex):
    async def refuse(method, params, original=codex._server.request):
        if method == "thread/realtime/start":
            codex._server.requests.append((method, params))
            codex._server.feed(
                params["threadId"],
                "thread/realtime/error",
                message="You've hit your usage limit. private account detail",
            )
            return {}
        return await original(method, params)

    codex._server.request = refuse
    with pytest.raises(VoiceError) as refused:
        await codex.create(None, SESSION, "offer-sdp")
    assert refused.value.metadata == {
        "allocation_status": "rejected",
        "reason": "usage_limit_reached",
    }
    assert "private" not in str(refused.value)
    assert codex._server.sent("thread/realtime/stop") == [{"threadId": "thread-1"}]
    assert "thread-1" not in codex._server.queues


async def _events(connection, count):
    events = []
    async with asyncio.timeout(1):
        while len(events) < count:
            events.append(json.loads(await anext(connection)))
    return events


async def test_connection_maps_the_call_and_interrupts_codex_agent_turns(codex):
    server = codex._server
    connection = await codex.attach(None, "thread-1")
    server.feed("thread-1", "thread/realtime/started", version="v3")
    server.feed("thread-1", "turn/started", turn={"id": "turn-7"})
    server.feed("thread-1", "thread/realtime/transcript/delta", role="user", delta="Book it")
    server.feed("thread-1", "thread/realtime/transcript/done", role="user", text="Book it")
    server.feed(
        "thread-1",
        "thread/realtime/itemAdded",
        item={"type": "handoff_request", "handoff_id": "h1", "input_transcript": "Book a room"},
    )
    server.feed("thread-1", "thread/realtime/itemAdded", item={"type": "other"})
    server.feed("thread-1", "thread/realtime/error", message="secret provider text")
    events = await _events(connection, 5)
    assert events == [
        {"type": "session.started"},
        {"type": "session.input_transcript.delta", "delta": "Book it"},
        {"type": "session.transcript.done", "role": "user", "text": "Book it"},
        {
            "type": "session.delegation.created",
            "delegation": {"id": "h1", "target": "client", "input": "Book a room"},
        },
        {"type": "error", "error": {"code": "codex_realtime_error"}},
    ]
    await until(lambda: server.sent("turn/interrupt"))
    assert server.sent("turn/interrupt") == [{"threadId": "thread-1", "turnId": "turn-7"}]

    await connection.send(
        json.dumps({"type": "session.commentary.append", "event_id": "c1", "content": "Booked."})
    )
    assert await _events(connection, 1) == [
        {"type": "session.commentary.appended", "client_event_id": "c1"}
    ]
    assert server.sent("thread/realtime/appendSpeech") == [
        {"threadId": "thread-1", "text": "Booked."}
    ]
    server.failures.add("thread/realtime/appendSpeech")
    await connection.send(
        json.dumps({"type": "session.commentary.append", "event_id": "c2", "content": "x"})
    )
    assert await _events(connection, 1) == [
        {"type": "error", "error": {"code": "codex_realtime_error", "client_event_id": "c2"}}
    ]

    await connection.send(json.dumps({"type": "session.close"}))
    [closed] = await _events(connection, 1)
    assert closed["type"] == "session.closed"
    assert closed["reason"] is None  # the runtime's own stop keeps the service's reason
    assert set(closed["usage"]) == {"duration_seconds"}
    await connection.close()


async def test_a_wrong_realtime_version_ends_the_call(codex):
    connection = await codex.attach(None, "thread-1")
    codex._server.feed("thread-1", "thread/realtime/started", version="v1")
    events = await _events(connection, 2)
    assert events[0] == {"type": "error", "error": {"code": "codex_version_mismatch"}}
    assert events[1]["type"] == "session.closed"
    assert events[1]["reason"] == "version_mismatch"
    await connection.close()


async def test_the_usage_guard_stops_a_running_call(codex):
    codex.usage_check_seconds = 0.01
    codex._server.results["account/rateLimits/read"] = _limits(ordinaryUsageAllowed=False)
    connection = await codex.attach(None, "thread-1")
    events = await _events(connection, 2)
    assert events[0] == {"type": "error", "error": {"code": "usage_not_allowed"}}
    assert events[1]["type"] == "session.closed"
    assert events[1]["reason"] == "usage_guard"
    await connection.close()


async def test_voice_service_runs_codex_calls_without_a_key_and_delegates(codex, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = Backend()
    service = VoiceService(
        VoiceConfig(enabled=True, provider="codex", close_timeout_seconds=0.05),
        backend,
        transport=codex,
    )
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        [realtime] = codex._server.sent("thread/realtime/start")
        assert realtime["voice"] == "cove"
        call_id = created["call_id"]
        codex._server.feed(
            "thread-1", "thread/realtime/transcript/done", role="assistant", text="Hi"
        )
        codex._server.feed(
            "thread-1",
            "thread/realtime/itemAdded",
            item={"type": "handoff_request", "handoff_id": "h1", "input_transcript": "Book a room"},
        )
        await until(lambda: codex._server.sent("thread/realtime/appendSpeech"))
        assert "Book a room" in backend.requests[0].content
        assert codex._server.sent("thread/realtime/appendSpeech")[0]["text"] == "Confirmed"
        record = await service.get(call_id, OWNER)
        assert record["model"] == "codex-realtime-v3"
        assert record["last_activity_at"] is not None
        assert record["delegations"]["h1"]["status"] in ("result_sent", "result_accepted")
        events = [event for _, event in service._calls[call_id].events]
        assert {"role": "assistant", "text": "Hi"} in [
            e["data"] for e in events if e["event"] == "transcript_done"
        ]
        status = await service.health_check()
        assert status["provider"] == "codex"
        assert status["codex"] == {"version": "0.157.1", "compatible": True, "missing": []}
        assert "cove" in status["voices"]
        assert (await service.usage())["allowed"] is True
        closed = await service.close(call_id, OWNER)
        assert closed["finalized"] is True
    finally:
        await service.stop()


async def test_usage_is_reported_only_for_the_codex_provider():
    service = VoiceService(VoiceConfig(enabled=True), Backend(), transport=object())
    with pytest.raises(VoiceError) as live:
        await service.usage()
    assert live.value.status_code == 404


def test_codex_voice_config_defaults_and_validates_the_voice():
    assert VoiceConfig().voice == "marin"
    assert VoiceConfig(provider="codex").voice == "cove"
    with pytest.raises(ValueError, match="codex provider"):
        VoiceConfig(provider="codex", voice="marin")


FAKE_CODEX = """
import json, os, sys
refused = False
for line in sys.stdin:
    message = json.loads(line)
    method, ident = message.get("method"), message.get("id")
    if method == "initialize":
        # A server request the client must refuse rather than leave hanging.
        request = {"jsonrpc": "2.0", "id": "srv-1", "method": "item/tool/call", "params": {}}
        print(json.dumps(request), flush=True)
        print(json.dumps({"id": ident, "result": {"userAgent": "codex/0.157.1"}}), flush=True)
    elif ident == "srv-1":
        refused = "error" in message
    elif method == "account/read":
        leaked = [k for k in ("OPENAI_API_KEY", "CODEX_API_KEY") if k in os.environ]
        result = {"account": {"type": "chatgpt"}, "leaked": leaked, "refused": refused}
        print(json.dumps({"id": ident, "result": result}), flush=True)
    elif ident is not None:
        print(json.dumps({"id": ident, "result": {"echo": method}}), flush=True)
"""


async def test_app_server_client_strips_api_keys_and_refuses_server_requests(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex.py"
    script.write_text(FAKE_CODEX)
    command = tmp_path / "codex"
    command.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    command.chmod(command.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-the-cli")
    monkeypatch.setenv("CODEX_API_KEY", "must-not-reach-the-cli")
    server = _AppServer(str(command), 5)
    try:
        await server.ensure_started()
        account = await server.request("account/read", {})
        assert account["leaked"] == []
        assert account["refused"] is True
        assert (await server.request("anything", {}))["echo"] == "anything"
    finally:
        await server.stop()
    assert os.environ["OPENAI_API_KEY"] == "must-not-reach-the-cli"


async def test_routes_work_under_a_host_mount(monkeypatch):
    import httpx
    from fastapi import FastAPI

    from assistant_runtime.app.access.deps import get_principal
    from assistant_runtime.app.routes.voice import router
    from tests.voice_helpers import Transport

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    service = VoiceService(
        VoiceConfig(enabled=True, close_timeout_seconds=0.05), Backend(), transport=Transport()
    )
    await service.start()
    runtime = FastAPI()
    runtime.include_router(router, prefix="/api")
    runtime.state.voice_service = service
    runtime.dependency_overrides[get_principal] = lambda: OWNER
    host = FastAPI()
    host.mount("/runtime", runtime)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=host), base_url="http://host"
        ) as client:
            created = await client.post(
                "/runtime/api/voice/calls", json={"session_id": "mounted", "sdp": "offer"}
            )
            assert created.status_code == 201
            call_id = created.json()["call_id"]
            assert created.json()["events_url"] == f"/runtime/api/voice/calls/{call_id}/events"
            assert (await client.get("/runtime/api/voice/usage")).status_code == 404
            assert (
                await client.post(f"/runtime/api/voice/calls/{call_id}/close")
            ).status_code == 200
    finally:
        await service.stop()


async def test_a_lost_app_server_leaves_the_call_interrupted_not_finalized(codex, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        codex._server.feed("thread-1", "thread/realtime/closed", reason="app_server_exit")
        call = service._calls[created["call_id"]]
        await until(call.done.is_set)
        record = await service.get(created["call_id"], OWNER)
        assert (record["status"], record["finalized"]) == ("interrupted", False)
        assert record["reason"] == "connection_lost"
    finally:
        await service.stop()


async def test_facts_reach_a_codex_call_quietly_or_as_speech_and_are_acknowledged(
    codex, monkeypatch
):
    from pydantic import ValidationError

    from assistant_runtime.app.voice.models import VoiceContext

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        call_id = created["call_id"]
        [realtime] = codex._server.sent("thread/realtime/start")
        assert "do not read a fact aloud" in realtime["prompt"]

        quiet = await service.update_context(call_id, VoiceContext(fact="Sent the message."), OWNER)
        assert quiet == {"updated": True, "fact": {"accepted": True, "speak": False}}
        assert codex._server.sent("thread/realtime/appendText") == [
            {"threadId": "thread-1", "text": "Sent the message.", "role": "developer"}
        ]
        with pytest.raises(VoiceError) as limited:
            await service.update_context(call_id, VoiceContext(fact="Too soon."), OWNER)
        assert limited.value.status_code == 429

        service._calls[call_id].last_fact_at -= 1
        spoken = await service.update_context(
            call_id, VoiceContext(fact="The review finished.", speak=True), OWNER
        )
        assert spoken["fact"]["speak"] is True
        assert codex._server.sent("thread/realtime/appendSpeech") == [
            {"threadId": "thread-1", "text": "The review finished."}
        ]

        service._calls[call_id].last_fact_at -= 1
        codex._server.failures.add("thread/realtime/appendText")
        with pytest.raises(VoiceError) as refused:
            await service.update_context(call_id, VoiceContext(fact="Lost."), OWNER)
        assert refused.value.status_code == 502
    finally:
        await service.stop()

    for body in ({}, {"fact": " "}, {"fact": "x" * 401}, {"host_context": {}, "speak": True}):
        with pytest.raises(ValidationError):
            VoiceContext.model_validate(body)


async def test_the_live_provider_refuses_facts_through_the_runtime(monkeypatch):
    from assistant_runtime.app.voice.models import VoiceContext
    from tests.voice_helpers import Transport

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-never-sent")
    service = VoiceService(
        VoiceConfig(enabled=True, close_timeout_seconds=0.05), Backend(), transport=Transport()
    )
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer"), OWNER)
        with pytest.raises(VoiceError) as refused:
            await service.update_context(created["call_id"], VoiceContext(fact="Done."), OWNER)
        assert refused.value.status_code == 409
    finally:
        await service.stop()
