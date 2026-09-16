"""Tests for OAuth routes."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.oauth import router
from assistant_runtime.services.oauth.interface import AuthSource, AuthStatus, DeviceCodeStatus


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
        email="assistant@example.com",
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

        assert data["email"] == "assistant@example.com"

    @pytest.mark.asyncio
    async def test_status_includes_source(self, client):
        response = await client.get("/api/oauth/openai/status")
        assert response.status_code == 200
        data = response.json()
        assert data["connected"] is True
        assert data["status"] == "authorized"
        assert data["source"] == "codex_cli"


@pytest.mark.parametrize("endpoint", ["codex-cli/sync", "device-code"])
async def test_disabled_oauth_returns_actionable_503_without_reading_auth_file(endpoint):
    from unittest.mock import patch

    from assistant_runtime.main import create_app
    from assistant_runtime.services.oauth.config import OAuthConfig
    from assistant_runtime.services.oauth.interface import OAuthService

    service = OAuthService(OAuthConfig())
    app = create_app()
    app.state.oauth_service = service
    with patch.object(service, "_read_codex_cli_auth") as read_auth:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/api/oauth/openai/{endpoint}")
    assert response.status_code == 503
    assert response.json()["type"] == "OAuthNotConfiguredError"
    assert "OAUTH__ENCRYPTION_KEY" in response.json()["error"]
    read_auth.assert_not_called()


@pytest.mark.parametrize("contents", [None, b"[]", b"null", b"invalid json", b"\xff"])
async def test_invalid_auth_file_returns_distinct_400(tmp_path, contents):
    from cryptography.fernet import Fernet

    from assistant_runtime.main import create_app
    from assistant_runtime.services.oauth.config import OAuthConfig
    from assistant_runtime.services.oauth.interface import OAuthService

    auth_file = tmp_path / "auth.json"
    if contents is not None:
        auth_file.write_bytes(contents)
    service = OAuthService(
        OAuthConfig(
            encryption_key=Fernet.generate_key().decode(),
            codex_auth_file=str(auth_file),
        )
    )
    await service.start()
    try:
        app = create_app()
        app.state.oauth_service = service
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/oauth/openai/codex-cli/sync")
        assert response.status_code == 400
        assert response.json()["type"] == "OAuthCodexSyncError"
    finally:
        await service.stop()


@pytest.mark.parametrize("deleted", [False, True])
async def test_disconnect_exposes_persisted_deletion_outcome(deleted):
    service = SimpleNamespace(disconnect=AsyncMock(return_value=deleted))
    app = _make_app(service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.delete("/api/oauth/openai")
    assert response.status_code == 200
    assert response.json() == {"status": "disconnected", "persisted_deleted": deleted}
