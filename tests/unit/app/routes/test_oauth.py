"""Tests for OAuth routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.routes.oauth import router
from lovely_assistant.services.oauth.interface import AuthSource, AuthStatus, DeviceCodeStatus


def _make_app(oauth_service) -> FastAPI:
    app = FastAPI()
    app.state.oauth_service = oauth_service
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
async def client():
    status = AuthStatus(
        connected=True,
        status=DeviceCodeStatus.AUTHORIZED,
        source=AuthSource.CODEX_CLI,
        email="jarvis@example.com",
        api_key_preview="sk-...1234",
    )
    oauth_service = SimpleNamespace(
        initiate_device_code=AsyncMock(
            return_value=SimpleNamespace(model_dump=lambda: {"user_code": "ABCD-1234"})
        ),
        sync_from_codex_cli=AsyncMock(return_value=status),
        get_device_code_status=lambda: status,
        disconnect=AsyncMock(),
    )
    app = _make_app(oauth_service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestOAuthRoutes:
    @pytest.mark.asyncio
    async def test_sync_codex_cli_returns_status(self, client):
        response = await client.post("/api/oauth/openai/codex-cli/sync")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["source"] == "codex_cli"
        assert data["email"] == "jarvis@example.com"

    @pytest.mark.asyncio
    async def test_status_includes_source(self, client):
        response = await client.get("/api/oauth/openai/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["status"] == "authorized"
        assert data["source"] == "codex_cli"
