"""PROTOTYPE: the Codex realtime transport forwards the offer and maps events."""

import asyncio
import json

import pytest

from assistant_runtime.app.voice import _codex_realtime
from assistant_runtime.app.voice._codex_realtime import CodexRealtimeTransport, _translate
from assistant_runtime.app.voice.exceptions import VoiceError


def test_events_map_to_the_sideband_shape():
    assert _translate({"method": "thread/realtime/started", "params": {}}) == {
        "type": "session.started"
    }
    assert _translate(
        {"method": "thread/realtime/transcript/delta", "params": {"role": "user", "delta": "hi"}}
    ) == {"type": "session.input_transcript.delta", "delta": "hi"}
    assert _translate(
        {
            "method": "thread/realtime/transcript/delta",
            "params": {"role": "assistant", "delta": "yo"},
        }
    ) == {"type": "session.output_transcript.delta", "delta": "yo"}
    assert _translate({"method": "thread/realtime/closed", "params": {"reason": "done"}}) == {
        "type": "session.closed",
        "reason": "done",
    }
    assert _translate({"method": "thread/realtime/outputAudio/delta", "params": {}}) is None


class FakeServer:
    def __init__(self, notifications):
        self.requests = []
        self.queue = asyncio.Queue()
        self.notifications = notifications
        self.problem = None

    async def ensure_started(self):
        pass

    async def usage_problem(self):
        return self.problem

    async def request(self, method, params):
        self.requests.append((method, params))
        if method == "thread/start":
            return {"thread": {"id": "t1"}}
        if method == "thread/realtime/start":
            for message in self.notifications:
                self.queue.put_nowait(message)
        return {}

    def subscribe(self, thread_id):
        return self.queue

    def unsubscribe(self, thread_id):
        pass


async def test_create_forwards_the_offer_and_returns_the_answer():
    transport = CodexRealtimeTransport(timeout=1)
    transport._server = FakeServer(
        [
            {"method": "thread/realtime/started", "params": {"threadId": "t1"}},
            {"method": "thread/realtime/sdp", "params": {"threadId": "t1", "sdp": "answer"}},
        ]
    )
    thread_id, answer = await transport.create(
        None, {"instructions": "be brief", "audio": {"output": {"voice": "juniper"}}}, "offer"
    )
    assert (thread_id, answer) == ("t1", "answer")
    method, params = transport._server.requests[-1]
    assert method == "thread/realtime/start"
    assert params["transport"] == {"type": "webrtc", "sdp": "offer"}
    assert params["version"] == "v3"
    assert params["clientManagedHandoffs"] is True
    assert params["voice"] == "juniper"
    connection = await transport.attach(None, thread_id)
    assert json.loads(await connection.__anext__()) == {"type": "session.started"}


async def test_a_refused_session_is_reported():
    transport = CodexRealtimeTransport(timeout=1)
    transport._server = FakeServer(
        [{"method": "thread/realtime/error", "params": {"threadId": "t1", "message": "no"}}]
    )
    with pytest.raises(VoiceError):
        await transport.create(None, {"instructions": "x"}, "offer")


def test_api_key_variables_are_removed_from_the_cli_environment(monkeypatch):
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured.update(kwargs)
        raise FileNotFoundError

    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(VoiceError):
        asyncio.run(_codex_realtime._AppServer(1).ensure_started())
    assert "OPENAI_API_KEY" not in captured["env"]


async def test_a_voice_the_provider_rejects_is_left_to_its_default():
    transport = CodexRealtimeTransport(timeout=1)
    transport._server = FakeServer(
        [{"method": "thread/realtime/sdp", "params": {"threadId": "t1", "sdp": "answer"}}]
    )
    await transport.create(
        None, {"instructions": "x", "audio": {"output": {"voice": "marin"}}}, "offer"
    )
    assert "voice" not in transport._server.requests[-1][1]


async def test_history_is_forwarded_as_initial_items():
    transport = CodexRealtimeTransport(timeout=1)
    transport._server = FakeServer(
        [{"method": "thread/realtime/sdp", "params": {"threadId": "t1", "sdp": "answer"}}]
    )
    seed = [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
    ]
    await transport.create(None, {"instructions": "x", "input": seed}, "offer")
    assert transport._server.requests[-1][1]["initialItems"] == [
        {"role": "user", "text": "hi"},
        {"role": "assistant", "text": "hello"},
    ]


async def test_the_usage_guard_refuses_a_call():
    transport = CodexRealtimeTransport(timeout=1)
    transport._server = FakeServer([])
    transport._server.problem = "the account has credits that could be charged"
    with pytest.raises(VoiceError, match="usage guard"):
        await transport.create(None, {"instructions": "x"}, "offer")
    assert transport._server.requests == []


async def test_usage_guard_reasons():
    server = _codex_realtime._AppServer(1)

    async def limits(method, params):
        return {"rateLimitsByLimitId": {"codex": snapshot}, "ordinaryUsageAllowed": True}

    server.request = limits
    snapshot = {
        "credits": {"hasCredits": False, "unlimited": False, "balance": "0"},
        "primary": {"usedPercent": 91},
    }
    assert await server.usage_problem() is None
    snapshot["primary"]["usedPercent"] = 97
    assert "97%" in await server.usage_problem()
    snapshot["primary"]["usedPercent"] = 50
    snapshot["credits"]["balance"] = "5"
    assert "credits" in await server.usage_problem()
