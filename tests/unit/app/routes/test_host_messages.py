"""POST /sessions/{id}/messages: a host card, refused while the session is busy."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.routes.sessions import router
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError

CARD = {"kind": "component", "type": "proposal", "data": {"version": 3}}


def _app(append) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.streaming_service = type("Streaming", (), {"append_host_message": append})()
    return app


async def test_a_card_is_appended_and_shown_as_a_host_message():
    record = {
        "id": "h1",
        "parent_id": "a1",
        "role": "host",
        "content": "Shown: a proposal",
        "segments": [CARD],
        "created_at": datetime(2026, 9, 27, tzinfo=UTC),
    }
    append = AsyncMock(return_value=record)
    async with AsyncClient(transport=ASGITransport(app=_app(append)), base_url="http://t") as c:
        response = await c.post(
            "/sessions/s1/messages", json={"content": "Shown: a proposal", "segments": [CARD]}
        )
    assert response.status_code == 201
    body = response.json()
    assert (body["role"], body["text"], body["segments"]) == ("host", "Shown: a proposal", [CARD])
    assert append.await_args.kwargs["segments"] == [CARD]


async def test_busy_unknown_profile_and_malformed_requests_are_refused():
    busy = AsyncMock(side_effect=SessionError("A turn or a host action is in progress"))
    unknown = AsyncMock(side_effect=UnknownProfileError("No profile 'nope'"))
    async with AsyncClient(transport=ASGITransport(app=_app(busy)), base_url="http://t") as c:
        assert (await c.post("/sessions/s1/messages", json={"segments": [CARD]})).status_code == 409
        text_only = await c.post(
            "/sessions/s1/messages", json={"segments": [{"kind": "text", "text": "hi"}]}
        )
        none = await c.post("/sessions/s1/messages", json={"content": "x", "segments": []})
        too_long = await c.post(f"/sessions/{'s' * 65}/messages", json={"segments": [CARD]})
    async with AsyncClient(transport=ASGITransport(app=_app(unknown)), base_url="http://t") as c:
        body = {"segments": [CARD], "profile": "nope"}
        assert (await c.post("/sessions/s1/messages", json=body)).status_code == 404
    assert (text_only.status_code, none.status_code, too_long.status_code) == (422, 422, 422)
