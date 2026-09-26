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
    started = (
        {"threadId": {}} if drop_param == "started.version" else {"threadId": {}, "version": {}}
    )
    (directory / "v2" / "ThreadRealtimeStartedNotification.json").write_text(
        json.dumps({"properties": started})
    )
    thread = ["approvalPolicy", "cwd", "developerInstructions", "ephemeral", "sandbox"]
    (directory / "v2" / "ThreadStartParams.json").write_text(
        json.dumps({"properties": {p: {} for p in thread if f"thread.{p}" != drop_param}})
    )
    (directory / "v2" / "ThreadRealtimeAppendTextParams.json").write_text(
        json.dumps({"properties": {"threadId": {}, "text": {}, "role": {}}})
    )
    snapshot = {"credits": {}, "spendControlReached": {}, "primary": {}}
    if drop_param == "snapshot.credits":
        del snapshot["credits"]
    (directory / "v2" / "GetAccountRateLimitsResponse.json").write_text(
        json.dumps({"definitions": {"RateLimitSnapshot": {"properties": snapshot}}})
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
        (
            "no_started_version",
            {"drop_param": "started.version"},
            ["thread/realtime/started.version"],
        ),
        ("no_ephemeral", {"drop_param": "thread.ephemeral"}, ["thread/start.ephemeral"]),
        ("no_credits", {"drop_param": "snapshot.credits"}, ["account/rateLimits/read.credits"]),
    ):
        directory = tmp_path / name
        directory.mkdir()
        _schema(directory, **kwargs)
        assert missing_from_schema(directory) == missing
    assert "thread/start" in missing_from_schema(tmp_path / "absent")


class FakeServer:
    """Scripted app-server: records requests and feeds notifications per thread."""

    running = True
    login = "chatgpt"

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

    async def require_chatgpt(self):
        self.requests.append(("account/read", {}))
        if self.login != "chatgpt":
            raise VoiceError("Codex voice requires the Codex CLI to be signed in with ChatGPT", 503)

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
        # Like the real reader: only subscribed threads receive notifications.
        if thread_id in self.queues:
            self.queues[thread_id].put_nowait(
                {"method": method, "params": {"threadId": thread_id, **params}}
            )

    def sent(self, method):
        return [params for name, params in self.requests if name == method]


@pytest.fixture
def codex():
    transport = CodexTransport(
        "codex", 1, usage_ceiling_percent=97, usage_check_seconds=60, close_timeout=1
    )
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
    assert "thread-1" not in codex._server.queues
    await until(lambda: codex._server.sent("thread/realtime/stop"))
    assert codex._server.sent("thread/realtime/stop") == [{"threadId": "thread-1"}]


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
    assert events[1]["reason"] == "codex_version_mismatch"
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


def _fake_codex(tmp_path, source):
    script = tmp_path / "fake_codex.py"
    script.write_text(source)
    command = tmp_path / "codex"
    command.write_text(f"#!/bin/sh\nexec {sys.executable} {script}\n")
    command.chmod(command.stat().st_mode | stat.S_IEXEC)
    return str(command)


WRONG_LOGIN = """
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if message.get("method") == "initialize":
        print(json.dumps({"id": message["id"], "result": {}}), flush=True)
    elif message.get("method") == "account/read":
        print(json.dumps({"id": message["id"], "result": {"account": {"type": "apiKey"}}}), flush=True)
"""

OVERSIZED = """
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    method, ident = message.get("method"), message.get("id")
    if method == "initialize":
        print(json.dumps({"id": ident, "result": {}}), flush=True)
    elif method == "account/read":
        print(json.dumps({"id": ident, "result": {"account": {"type": "chatgpt"}}}), flush=True)
    elif method == "boom":
        sys.stdout.write("x" * (2**24 + 10) + "\\n")
        sys.stdout.flush()
"""


async def test_a_server_on_another_login_is_never_reused(tmp_path):
    server = _AppServer(_fake_codex(tmp_path, WRONG_LOGIN), 5)
    try:
        for _ in range(2):  # the second call checks a new server again
            with pytest.raises(VoiceError, match="ChatGPT"):
                await server.ensure_started()
            assert server._link is None
    finally:
        await server.stop()


async def test_a_failed_read_ends_the_server_and_releases_waiters(tmp_path):
    server = _AppServer(_fake_codex(tmp_path, OVERSIZED), 5)
    try:
        await server.ensure_started()
        calls = server.subscribe("thread-1")
        with pytest.raises(VoiceError, match="exited"):
            await server.request("boom", {})
        closed = await asyncio.wait_for(calls.get(), 5)
        assert closed["params"]["reason"] == "app_server_exit"
        assert not server.running  # the next call starts a new server
    finally:
        await server.stop()


async def test_a_cli_that_cannot_run_is_checked_again(tmp_path):
    transport = CodexTransport(
        str(tmp_path / "missing-codex"),
        1,
        usage_ceiling_percent=97,
        usage_check_seconds=60,
        close_timeout=1,
    )
    result = await transport.compatibility()
    assert result == {"version": None, "compatible": False, "missing": ["codex_cli"]}
    assert transport.checked() is None


async def test_a_cancelled_create_still_stops_the_session(codex):
    async def silent(method, params, original=codex._server.request):
        if method == "thread/realtime/start":
            codex._server.requests.append((method, params))
            await asyncio.sleep(10)
        return await original(method, params)

    codex._server.request = silent
    creating = asyncio.create_task(codex.create(None, SESSION, "offer-sdp"))
    await until(lambda: codex._server.sent("thread/realtime/start"))
    creating.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creating
    await until(lambda: codex._server.sent("thread/realtime/stop"))
    workdir = codex._workdir.name
    await codex.stop()
    assert not os.path.exists(workdir)


async def test_a_slow_stop_does_not_block_the_sender_and_close_waits_for_it(codex):
    finished = asyncio.Event()

    async def slow(method, params, original=codex._server.request):
        if method == "thread/realtime/stop":
            codex._server.requests.append((method, params))
            await asyncio.sleep(0.1)
            finished.set()
            return {}
        return await original(method, params)

    codex._server.request = slow
    connection = await codex.attach(None, "thread-1")
    async with asyncio.timeout(0.05):  # far shorter than the stop takes
        await connection.send(json.dumps({"type": "session.close"}))
    assert not finished.is_set()
    await connection.close()
    assert finished.is_set()
    assert len(codex._server.sent("thread/realtime/stop")) == 1


async def test_only_repeated_unreadable_usage_ends_a_call(codex):
    codex.usage_check_seconds = 0.01
    reads = 0

    async def flaky(method, params, original=codex._server.request):
        nonlocal reads
        if method == "account/rateLimits/read":
            reads += 1
            if reads in (1, 2, 4, 5, 6):
                raise VoiceError("Codex account/rateLimits/read failed", 502)
        return await original(method, params)

    codex._server.request = flaky
    connection = await codex.attach(None, "thread-1")
    events = await _events(connection, 2)
    assert reads == 6  # two failures were tolerated; the third in a row stopped it
    assert events[0] == {"type": "error", "error": {"code": "usage_unreadable"}}
    assert events[1]["reason"] == "usage_guard"
    await connection.close()


async def test_a_refused_fact_changes_nothing(codex, monkeypatch):
    from assistant_runtime.app.voice.models import VoiceContext
    from assistant_runtime.host_context import HostContext

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        call = service._calls[created["call_id"]]
        codex._server.failures.add("thread/realtime/appendText")
        update = VoiceContext(fact="Lost.", host_context=HostContext(background={"page": "new"}))
        with pytest.raises(VoiceError):
            await service.update_context(call.id, update, OWNER)
        assert call.offer.host_context is None
        assert call.last_fact_at is None  # the next fact is not rate-limited
    finally:
        await service.stop()


async def test_status_reports_the_background_check_without_running_it(monkeypatch):
    transport = CodexTransport(
        "codex", 1, usage_ceiling_percent=97, usage_check_seconds=60, close_timeout=1
    )
    transport._server = FakeServer()
    gate = asyncio.Event()

    async def blocked():
        await gate.wait()
        return {"version": "0.157.1", "compatible": True, "missing": []}

    transport._check = blocked
    service = VoiceService(
        VoiceConfig(enabled=True, provider="codex"), Backend(), transport=transport
    )
    await service.start()
    try:
        assert (await service.health_check())["codex"] is None
        gate.set()
        await until(lambda: transport.checked() is not None)
        assert (await service.health_check())["codex"]["compatible"] is True
    finally:
        await service.stop()


def test_a_zero_balance_in_any_format_is_not_spendable_credit():
    for balance, allowed in (
        ("0.00", True),
        ("0", True),
        (None, True),
        ("1.50", False),
        ("n/a", False),
    ):
        report = usage_report(_limits(snapshot={"credits": {"balance": balance}}), 97)
        assert report["allowed"] is allowed, balance


def test_one_unusual_schema_variant_does_not_hide_the_others(tmp_path):
    _schema(tmp_path)
    requests = json.loads((tmp_path / "ClientRequest.json").read_text())
    requests["oneOf"].append({"properties": {"method": {"const": "future/method"}}})
    requests["oneOf"].append({"properties": {"params": {}}})
    (tmp_path / "ClientRequest.json").write_text(json.dumps(requests))
    assert missing_from_schema(tmp_path) == []


async def test_an_empty_answer_stops_the_started_session(codex):
    async def empty(method, params, original=codex._server.request):
        if method == "thread/realtime/start":
            codex._server.requests.append((method, params))
            codex._server.feed(params["threadId"], "thread/realtime/sdp", sdp="")
            return {}
        return await original(method, params)

    codex._server.request = empty
    with pytest.raises(VoiceError) as failed:
        await codex.create(None, SESSION, "offer-sdp")
    assert failed.value.metadata == {"allocation_status": "unknown"}
    await until(lambda: codex._server.sent("thread/realtime/stop"))


async def test_usage_is_not_read_while_the_service_is_stopped(codex):
    started = 0

    async def counted():
        nonlocal started
        started += 1

    codex._server.ensure_started = counted
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    with pytest.raises(VoiceError) as stopped:
        await service.usage()
    assert stopped.value.status_code == 503
    assert started == 0


async def test_a_servers_cleanup_ends_its_own_calls_even_after_it_was_replaced():
    from types import SimpleNamespace

    from assistant_runtime.app.voice._codex import _Link

    server = _AppServer("codex", 1)
    old_output = asyncio.StreamReader()
    old = _Link(SimpleNamespace(stdout=old_output, returncode=0))
    new = _Link(SimpleNamespace(returncode=None))
    server._link = new  # a newer server took over before the old one's cleanup
    loop = asyncio.get_running_loop()
    old_waiting, new_waiting = loop.create_future(), loop.create_future()
    old.pending[1], new.pending[2] = old_waiting, new_waiting
    old_call = old.threads.setdefault("thread-1", asyncio.Queue())
    new_call = new.threads.setdefault("thread-2", asyncio.Queue())
    reading = asyncio.create_task(server._read(old))
    old_output.feed_eof()
    await reading
    assert (await old_call.get())["params"]["reason"] == "app_server_exit"
    assert isinstance(old_waiting.exception(), VoiceError)
    assert not new_waiting.done()
    assert new_call.empty()


async def test_concurrent_facts_respect_the_per_second_limit(codex, monkeypatch):
    from assistant_runtime.app.voice.models import VoiceContext

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def slow(method, params, original=codex._server.request):
        if method == "thread/realtime/appendText":
            await asyncio.sleep(0.05)
        return await original(method, params)

    codex._server.request = slow
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        results = await asyncio.gather(
            *(
                service.update_context(created["call_id"], VoiceContext(fact=f"Fact {n}"), OWNER)
                for n in range(3)
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(r, dict) for r in results) == 1
        assert sorted(r.status_code for r in results if isinstance(r, VoiceError)) == [429, 429]
        assert len(codex._server.sent("thread/realtime/appendText")) == 1
    finally:
        await service.stop()


async def test_a_lost_server_is_not_restarted_to_read_usage(codex):
    codex.usage_check_seconds = 0.01
    codex._server.running = False
    started = 0

    async def counted():
        nonlocal started
        started += 1

    codex._server.ensure_started = counted
    connection = await codex.attach(None, "thread-1")
    events = await _events(connection, 1)
    assert events == [{"type": "error", "error": {"code": "usage_unreadable"}}]
    assert started == 0
    await connection.close()


async def test_a_session_the_provider_ended_is_not_stopped_again(codex):
    connection = await codex.attach(None, "thread-1")
    codex._server.feed("thread-1", "thread/realtime/closed", reason="transport_closed")
    [closed] = await _events(connection, 1)
    assert closed["reason"] == "transport_closed"
    await connection.close()
    assert codex._server.sent("thread/realtime/stop") == []


def test_an_unrecognised_usage_shape_is_not_allowed():
    for limits in ({}, {"rateLimits": None}, {"rateLimitsByLimitId": {"codex": "text"}}):
        report = usage_report(limits, 97)
        assert (report["allowed"], report["reason"]) == (False, "usage_unreadable"), limits


async def test_notifications_keep_their_order_around_the_answer(codex):
    async def burst(method, params, original=codex._server.request):
        if method == "thread/realtime/start":
            codex._server.requests.append((method, params))
            thread = params["threadId"]
            codex._server.feed(thread, "thread/realtime/started", version="v3")
            codex._server.feed(thread, "thread/realtime/sdp", sdp="answer-sdp")
            codex._server.feed(thread, "thread/realtime/closed", reason="transport_closed")
            return {}
        return await original(method, params)

    codex._server.request = burst
    thread, _ = await codex.create(None, SESSION, "offer-sdp")
    connection = await codex.attach(None, thread)
    events = await _events(connection, 2)
    assert [e["type"] for e in events] == ["session.started", "session.closed"]
    assert "duration_seconds" in events[1]["usage"]
    await connection.close()


async def test_a_transport_failure_while_sending_is_a_502(tmp_path):
    server = _AppServer(_fake_codex(tmp_path, OVERSIZED), 5)
    try:
        await server.ensure_started()
        server._link.proc.stdin.close()
        with pytest.raises(VoiceError) as failed:
            await server.request("thread/realtime/appendText", {})
        assert failed.value.status_code == 502
    finally:
        await server.stop()


async def test_a_superseded_delegation_leaves_no_request_text_behind(codex, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend = Backend()
    release = asyncio.Event()

    async def held(request):
        await release.wait()
        yield {"type": "final_response", "content": "Done", "model": "test", "error": False}

    backend.runner = held
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), backend, transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        call = service._calls[created["call_id"]]
        for ident in ("h1", "h2", "h3"):
            codex._server.feed(
                "thread-1",
                "thread/realtime/itemAdded",
                item={"type": "handoff_request", "handoff_id": ident, "input_transcript": ident},
            )
        await until(lambda: call.active_delegation == "h3")
        release.set()
        await until(lambda: call.delegations["h3"]["status"] in ("result_sent", "result_accepted"))
        assert call.inputs == {}
    finally:
        await service.stop()


async def test_every_call_rechecks_the_login_and_instructs_the_agent_not_to_act(codex):
    await codex.create(None, SESSION, "offer-sdp")
    [start] = codex._server.sent("thread/start")
    assert "Do not run commands" in start["developerInstructions"]

    codex._server.login = "apiKey"  # the CLI's login changed while its server ran
    with pytest.raises(VoiceError, match="ChatGPT"):
        await codex.create(None, SESSION, "offer-sdp")
    assert len(codex._server.sent("thread/start")) == 1


async def test_a_missing_cli_is_reported_as_unavailable_not_incompatible(codex):
    codex._compatibility = None

    async def unavailable():
        return {"version": None, "compatible": False, "missing": ["codex_cli"]}

    codex._check = unavailable
    with pytest.raises(VoiceError) as missing:
        await codex.create(None, SESSION, "offer-sdp")
    assert missing.value.metadata["reason"] == "codex_unavailable"
    assert "install it" in str(missing.value)


async def test_a_check_that_crashes_is_reported_and_retried(monkeypatch):
    from assistant_runtime.app.voice import _codex

    def broken(**kwargs):
        raise OSError("temporary directory unavailable")

    monkeypatch.setattr(_codex.tempfile, "TemporaryDirectory", broken)
    transport = CodexTransport(
        "codex", 1, usage_ceiling_percent=97, usage_check_seconds=60, close_timeout=1
    )
    assert (await transport.compatibility())["missing"] == ["codex_cli"]
    assert transport.checked() is None


async def test_a_failed_fact_does_not_undo_a_newer_facts_slot(codex, monkeypatch):
    from assistant_runtime.app.voice.models import VoiceContext

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    started = asyncio.Event()

    async def slow_failure(method, params, original=codex._server.request):
        if method == "thread/realtime/appendText":
            started.set()
            await asyncio.sleep(0.05)
            raise VoiceError("Codex thread/realtime/appendText failed", 502)
        return await original(method, params)

    codex._server.request = slow_failure
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        created = await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        call = service._calls[created["call_id"]]
        failing = asyncio.create_task(
            service.update_context(call.id, VoiceContext(fact="Lost."), OWNER)
        )
        await started.wait()
        newer = call.last_fact_at + 5  # another fact claimed the slot meanwhile
        call.last_fact_at = newer
        with pytest.raises(VoiceError):
            await failing
        assert call.last_fact_at == newer
    finally:
        await service.stop()


def test_a_moved_version_list_does_not_hide_the_start_parameters(tmp_path):
    _schema(tmp_path)
    path = tmp_path / "v2" / "ThreadRealtimeStartParams.json"
    start = json.loads(path.read_text())
    start["$defs"] = start.pop("definitions")
    path.write_text(json.dumps(start))
    assert missing_from_schema(tmp_path) == ["thread/realtime/start.version=v3"]


async def test_a_cli_that_cannot_run_is_retried_not_cached(tmp_path):
    command = _fake_codex(tmp_path, "import sys\nsys.exit(1)\n")
    transport = CodexTransport(
        command, 5, usage_ceiling_percent=97, usage_check_seconds=60, close_timeout=1
    )
    assert (await transport.compatibility())["missing"] == ["codex_cli"]
    assert transport.checked() is None


async def test_a_server_that_exits_before_the_answer_leaves_allocation_unknown(codex):
    async def dies(method, params, original=codex._server.request):
        if method == "thread/realtime/start":
            codex._server.requests.append((method, params))
            codex._server.feed(
                params["threadId"], "thread/realtime/closed", reason="app_server_exit"
            )
            return {}
        return await original(method, params)

    codex._server.request = dies
    with pytest.raises(VoiceError) as lost:
        await codex.create(None, SESSION, "offer-sdp")
    assert lost.value.metadata == {"allocation_status": "unknown"}


async def test_closing_a_call_releases_its_thread(codex):
    connection = await codex.attach(None, "thread-1")
    await connection.close()
    assert codex._server.sent("thread/unsubscribe") == [{"threadId": "thread-1"}]


def test_malformed_usage_values_read_as_unreadable():
    for snapshot in ({"credits": "none"}, {"primary": {"usedPercent": "12"}}):
        report = usage_report(_limits(snapshot=snapshot), 97)
        assert (report["allowed"], report["reason"]) == (False, "usage_unreadable"), snapshot


async def test_unreadable_usage_readings_get_the_same_tolerance_as_failed_reads(codex):
    codex.usage_check_seconds = 0.01
    reads = 0

    async def empty(method, params, original=codex._server.request):
        nonlocal reads
        if method == "account/rateLimits/read":
            reads += 1
            return {"rateLimitsByLimitId": {}}
        return await original(method, params)

    codex._server.request = empty
    connection = await codex.attach(None, "thread-1")
    events = await _events(connection, 1)
    assert events == [{"type": "error", "error": {"code": "usage_unreadable"}}]
    assert reads == 3
    await connection.close()


async def test_a_cli_that_runs_but_cannot_describe_its_protocol_is_incompatible(tmp_path):
    source = (
        "import sys\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('codex-cli 0.90.0')\n"
        "else:\n"
        "    sys.exit(2)\n"
    )
    script = tmp_path / "fake_codex.py"
    script.write_text(source)
    command = tmp_path / "codex"
    command.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    command.chmod(command.stat().st_mode | stat.S_IEXEC)
    transport = CodexTransport(
        str(command), 5, usage_ceiling_percent=97, usage_check_seconds=60, close_timeout=1
    )
    result = await transport.compatibility()
    assert result == {
        "version": "0.90.0",
        "compatible": False,
        "missing": ["app-server generate-json-schema"],
    }
    assert transport.checked() == result  # an old CLI stays old: no rerun per call


async def test_a_hung_server_cannot_hold_a_call_open_on_close(codex):
    codex.close_timeout = 0.05  # VOICE__CLOSE_TIMEOUT_SECONDS

    async def hung(method, params, original=codex._server.request):
        if method in ("thread/realtime/stop", "thread/unsubscribe"):
            await asyncio.sleep(10)
        return await original(method, params)

    codex._server.request = hung
    connection = await codex.attach(None, "thread-1")
    async with asyncio.timeout(1):
        await connection.close()


def test_absent_credits_are_no_credits_but_a_malformed_snapshot_set_is_unreadable():
    snapshot = dict(LIMITS["rateLimitsByLimitId"]["codex"])
    del snapshot["credits"]  # optional and nullable in the CLI's schema
    assert usage_report({"rateLimitsByLimitId": {"codex": snapshot}}, 97)["allowed"] is True
    snapshot["credits"] = None
    assert usage_report({"rateLimitsByLimitId": {"codex": snapshot}}, 97)["allowed"] is True
    report = usage_report({"rateLimitsByLimitId": ["not", "a", "mapping"]}, 97)
    assert report["reason"] == "usage_unreadable"


async def test_a_failed_interrupt_and_an_unknown_item_are_logged(codex):
    from loguru import logger

    messages = []
    sink = logger.add(messages.append, level="WARNING")
    codex._server.failures.add("turn/interrupt")
    try:
        connection = await codex.attach(None, "thread-1")
        codex._server.feed("thread-1", "turn/started", turn={"id": "turn-1"})
        codex._server.feed("thread-1", "thread/realtime/itemAdded", item={"type": "surprise"})
        codex._server.feed("thread-1", "thread/realtime/started", version="v3")
        await _events(connection, 1)
        await until(lambda: any("interrupt" in str(m) for m in messages))
        assert any("surprise" in str(m) for m in messages)
        await connection.close()
    finally:
        logger.remove(sink)


async def test_a_thread_start_without_an_id_is_rejected_cleanly(codex):
    codex._server.results["thread/start"] = {"threadId": "t1"}
    with pytest.raises(VoiceError) as bad:
        await codex.create(None, SESSION, "offer-sdp")
    assert bad.value.metadata == {"allocation_status": "rejected"}


async def test_a_session_created_but_never_attached_is_stopped(codex, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def fails(api_key, provider_id):
        raise RuntimeError("attach failed")

    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    codex.attach = fails
    await service.start()
    try:
        with pytest.raises(VoiceError):
            await service.create(VoiceOffer(session_id="talk", sdp="offer-sdp"), OWNER)
        await until(lambda: codex._server.sent("thread/realtime/stop"))
        assert codex._created == {}
        assert "thread-1" not in codex._server.queues
    finally:
        await service.stop()


async def test_final_transcripts_arrive_while_a_call_stops_but_not_past_its_limit(
    codex, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = VoiceService(VoiceConfig(enabled=True, provider="codex"), Backend(), transport=codex)
    await service.start()
    try:
        for reason, emitted in (("close_requested", True), ("transcript_limit", False)):
            created = await service.create(VoiceOffer(session_id=reason, sdp="offer-sdp"), OWNER)
            call = service._calls[created["call_id"]]
            thread = codex._server.sent("thread/start") and "thread-1"
            call.reason = reason
            call.stop_requested.set()
            codex._server.feed(thread, "thread/realtime/transcript/done", role="user", text="last")
            codex._server.feed(thread, "thread/realtime/closed", reason="requested")
            await until(call.done.is_set)
            finals = [e for _, e in call.events if e["event"] == "transcript_done"]
            assert bool(finals) is emitted, reason
    finally:
        await service.stop()


async def test_an_abandoned_session_is_stopped_and_released(codex):
    codex._server.subscribe("thread-9")
    codex.abandon("thread-9")
    await until(lambda: codex._server.sent("thread/unsubscribe"))
    assert codex._server.sent("thread/realtime/stop") == [{"threadId": "thread-9"}]
    assert codex._server.sent("thread/unsubscribe") == [{"threadId": "thread-9"}]
    assert "thread-9" not in codex._server.queues
