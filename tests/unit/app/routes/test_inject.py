"""Tests for the ingress endpoints: POST /assistant/inject and GET /assistant/sessions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.inject import router


def _make_app(ingress=None, assistant_service=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if ingress is not None:
        app.state.ingress_service = ingress
    if assistant_service is not None:
        app.state.assistant_service = assistant_service
    return app


@pytest.fixture
def ingress():
    service = MagicMock()
    service.deliver = AsyncMock(
        return_value={"status": "delivered", "session_id": "sess-1", "delivery": "queued"}
    )
    return service


class TestInjectMessage:
    async def test_delivers_through_the_ingress(self, ingress):
        app = _make_app(ingress=ingress)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject",
                json={"from": "leo", "via": "tmux", "message": "Done", "sessionId": "sess-1"},
            )
        assert response.status_code == 201
        assert response.json()["status"] == "delivered"
        assert ingress.deliver.await_args.kwargs == {
            "from_agent": "leo",
            "via": "tmux",
            "message": "Done",
            "session_id": "sess-1",
            "telegram_chat_id": None,
        }

    async def test_accepts_snake_case_and_telegram_binding(self, ingress):
        app = _make_app(ingress=ingress)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject",
                json={
                    "from_agent": "bot",
                    "via": "telegram",
                    "message": "hi",
                    "telegram_chat_id": "42",
                },
            )
        assert response.status_code == 201
        assert ingress.deliver.await_args.kwargs["telegram_chat_id"] == "42"

    async def test_queued_when_no_session(self, ingress):
        ingress.deliver = AsyncMock(return_value={"status": "queued", "inbox_id": "i-1"})
        app = _make_app(ingress=ingress)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject", json={"from": "bot", "via": "cron", "message": "tick"}
            )
        assert response.status_code == 201
        assert response.json() == {"status": "queued", "inbox_id": "i-1"}

    @pytest.mark.parametrize(
        "body",
        [{"via": "x", "message": "m"}, {"from": "a", "message": "m"}, {"from": "a", "via": "x"}],
    )
    async def test_missing_fields_are_422(self, ingress, body):
        app = _make_app(ingress=ingress)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/assistant/inject", json=body)
        assert response.status_code == 422

    async def test_delivery_failure_is_500(self, ingress):
        ingress.deliver = AsyncMock(side_effect=RuntimeError("down"))
        app = _make_app(ingress=ingress)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject", json={"from": "a", "via": "x", "message": "m"}
            )
        assert response.status_code == 500


class TestListSessionsForPeers:
    async def test_returns_peer_format(self):
        store = MagicMock()
        store.list_sessions = AsyncMock(
            return_value=[{"session_id": "s1", "title": "T", "turn_number": 3}]
        )
        assistant = MagicMock()
        assistant.get_session_store.return_value = store
        app = _make_app(assistant_service=assistant)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")
        assert response.json() == {
            "sessions": [{"id": "s1", "active": True, "title": "T", "turn_number": 3}]
        }

    async def test_empty_without_a_store(self):
        assistant = MagicMock()
        assistant.get_session_store.return_value = None
        app = _make_app(assistant_service=assistant)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")
        assert response.json() == {"sessions": []}

    async def test_empty_without_an_assistant_service(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")
        assert response.json() == {"sessions": []}
