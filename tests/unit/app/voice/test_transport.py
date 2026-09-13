"""HTTP protocol and optional persistence boundaries, with no external services."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from assistant_runtime.app.voice._persistence import VoicePersistence
from assistant_runtime.app.voice._transport import LiveTransport
from assistant_runtime.app.voice.exceptions import VoiceError


async def test_http_creation_uses_documented_live_endpoint_and_shape():
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(
            201,
            json={"session": {"id": "live_123"}, "transport": {"type": "webrtc", "sdp": "answer"}},
        )

    transport = LiveTransport(5)
    await transport.http.aclose()
    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        assert await transport.create(
            "secret", {"model": "gpt-live-1", "delegation": {"type": "client"}}, "offer"
        ) == ("live_123", "answer")
        assert str(requests[0].url) == "https://api.openai.com/v1/live/sessions"
        assert requests[0].headers["authorization"] == "Bearer secret"
        assert json.loads(requests[0].content)["transport"] == {"type": "webrtc", "sdp": "offer"}
    finally:
        await transport.stop()


@pytest.mark.parametrize("status", [401, 403, 429, 500])
async def test_creation_errors_are_not_retried_or_leaked(status):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "private upstream detail"})

    transport = LiveTransport(5)
    await transport.http.aclose()
    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(VoiceError) as result:
            await transport.create("secret", {}, "offer")
        assert len(requests) == 1
        assert "private" not in str(result.value)
        assert "secret" not in str(result.value)
    finally:
        await transport.stop()


async def test_unavailable_database_does_not_break_voice():
    db = SimpleNamespace(healthy=False, session_context=AsyncMock())
    persistence = VoicePersistence(db)
    assert await persistence.save({"call_id": "call"}) is False
    assert await persistence.load("call") is None
    db.session_context.assert_not_called()


async def test_checkpoint_upsert_and_load():
    class Database:
        healthy = True
        saved = {}

        def session_context(self):
            return self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def execute(self, statement):
            values = statement.compile().params
            self.saved[values["id"]] = values["snapshot"]

        async def get(self, model, key):
            return SimpleNamespace(snapshot=self.saved[key]) if key in self.saved else None

    db = Database()
    persistence = VoicePersistence(db)
    record = {
        "call_id": "call",
        "session_id": "thread",
        "usage": {"seconds": 7},
        "transcript": [{"role": "user", "delta": "hello"}],
    }
    assert await persistence.save(record)
    assert await persistence.load("call") == record
    record = {**record, "usage": {"seconds": 10}, "finalized": True}
    assert await persistence.save(record)
    assert (await persistence.load("call"))["usage"] == {"seconds": 10}
