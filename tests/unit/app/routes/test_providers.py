"""Tests for provider API key management routes (GET/PUT/DELETE /providers)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.providers import router

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_mock_db() -> MagicMock:
    """Create a mock DatabaseService with session_context."""
    mock_db = MagicMock()
    mock_db._healthy = True
    mock_session = AsyncMock()

    @asynccontextmanager
    async def fake_session_context():
        yield mock_session

    mock_db.session_context = fake_session_context
    mock_db._mock_session = mock_session
    return mock_db


def _make_mock_llm_service(
    *,
    provider_status: list[dict] | None = None,
) -> MagicMock:
    """Create a mock LlmService."""
    svc = MagicMock()
    svc.get_provider_status.return_value = provider_status or [
        {"provider": "anthropic", "configured": False, "source": None, "api_key_preview": None},
        {"provider": "openai", "configured": False, "source": None, "api_key_preview": None},
        {"provider": "google", "configured": False, "source": None, "api_key_preview": None},
        {"provider": "openrouter", "configured": False, "source": None, "api_key_preview": None},
    ]
    svc.reload_provider_key = AsyncMock()
    svc.remove_provider_key = AsyncMock()
    return svc


def _make_mock_oauth_service(
    *,
    configured: bool = True,
    fernet: Fernet | None = None,
    device_code_status: SimpleNamespace | None = None,
) -> SimpleNamespace:
    """Create a mock OAuthService."""
    if fernet is None and configured:
        fernet = Fernet(Fernet.generate_key())
    return SimpleNamespace(
        configured=configured,
        _fernet=fernet,
        get_device_code_status=lambda: (
            device_code_status
            or SimpleNamespace(
                connected=False,
                status="idle",
                email=None,
                source=None,
                expires_at=None,
            )
        ),
    )


def _make_app(
    *,
    llm_service: MagicMock | None = None,
    oauth_service: SimpleNamespace | None = None,
    db_service: MagicMock | None = None,
) -> FastAPI:
    """Create a minimal FastAPI app with provider routes."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    app.state.llm_service = llm_service or _make_mock_llm_service()
    if oauth_service is not None:
        app.state.oauth_service = oauth_service
    if db_service is not None:
        app.state.database_service = db_service
    return app


# ---------------------------------------------------------------------------
# GET /api/providers
# ---------------------------------------------------------------------------


class TestListProviders:
    @pytest.mark.asyncio
    async def test_list_providers_returns_all_known_providers(self):
        """GET /providers returns entries for all four known providers."""
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/api/providers")

        assert response.status_code == 200
        data = response.json()
        provider_names = {entry["provider"] for entry in data}
        assert provider_names == {"anthropic", "openai", "google", "openrouter"}

    @pytest.mark.asyncio
    async def test_list_providers_shows_configured_status(self):
        """When a provider has a key, configured=True with source."""
        llm = _make_mock_llm_service(
            provider_status=[
                {
                    "provider": "anthropic",
                    "configured": True,
                    "source": "environment",
                    "api_key_preview": "...t123",
                },
                {
                    "provider": "openai",
                    "configured": False,
                    "source": None,
                    "api_key_preview": None,
                },
                {
                    "provider": "google",
                    "configured": True,
                    "source": "database",
                    "api_key_preview": "...gkey",
                },
                {
                    "provider": "openrouter",
                    "configured": False,
                    "source": None,
                    "api_key_preview": None,
                },
            ]
        )
        app = _make_app(llm_service=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/api/providers")

        data = response.json()
        by_provider = {entry["provider"]: entry for entry in data}

        assert by_provider["anthropic"]["configured"] is True
        assert by_provider["anthropic"]["source"] == "environment"
        assert by_provider["openai"]["configured"] is False
        assert by_provider["google"]["configured"] is True
        assert by_provider["google"]["source"] == "database"
        assert by_provider["openrouter"]["configured"] is False

    @pytest.mark.asyncio
    async def test_list_providers_includes_oauth_status_for_openai(self):
        """OpenAI entry includes oauth info when oauth service is present."""
        oauth = _make_mock_oauth_service(
            device_code_status=SimpleNamespace(
                connected=True,
                status="authorized",
                email="user@example.com",
                source="codex_cli",
                expires_at=None,
            )
        )
        app = _make_app(oauth_service=oauth)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/api/providers")

        data = response.json()
        by_provider = {entry["provider"]: entry for entry in data}

        openai_entry = by_provider["openai"]
        assert openai_entry["oauth"] is not None
        assert openai_entry["oauth"]["connected"] is True
        assert openai_entry["oauth"]["status"] == "authorized"
        assert openai_entry["oauth"]["email"] == "user@example.com"
        assert openai_entry["oauth"]["source"] == "codex_cli"

        # OAuth connected but no direct API key -> configured via oauth
        assert openai_entry["configured"] is True
        assert openai_entry["source"] == "oauth"

        # Non-openai providers should have oauth=None
        assert by_provider["anthropic"]["oauth"] is None
        assert by_provider["google"]["oauth"] is None

    @pytest.mark.asyncio
    async def test_list_providers_includes_display_names(self):
        """Each entry includes a human-readable 'name' field."""
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/api/providers")

        data = response.json()
        by_provider = {entry["provider"]: entry for entry in data}
        assert by_provider["anthropic"]["name"] == "Anthropic"
        assert by_provider["openai"]["name"] == "OpenAI"
        assert by_provider["google"]["name"] == "Google"
        assert by_provider["openrouter"]["name"] == "OpenRouter"


# ---------------------------------------------------------------------------
# PUT / DELETE /api/providers/{provider}/api-key — through LlmService's key store
# ---------------------------------------------------------------------------


def _llm_with_key_store(*, store_error: Exception | None = None, deleted: bool = True):
    llm = _make_mock_llm_service()
    llm.store_provider_key = AsyncMock(side_effect=store_error)
    llm.delete_provider_key = AsyncMock(side_effect=store_error, return_value=deleted)
    return llm


class TestSetApiKey:
    @pytest.mark.asyncio
    async def test_set_api_key_stores_through_the_service(self):
        llm = _llm_with_key_store()
        app = _make_app(llm_service=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.put(
                "/api/providers/anthropic/api-key", json={"api_key": "sk-ant-test-key-1234"}
            )
        assert response.status_code == 200
        assert response.json() == {
            "provider": "anthropic",
            "status": "configured",
            "api_key_preview": "...1234",
        }
        llm.store_provider_key.assert_awaited_once_with("anthropic", "sk-ant-test-key-1234")

    @pytest.mark.asyncio
    async def test_set_api_key_rejects_unknown_provider(self):
        llm = _llm_with_key_store()
        app = _make_app(llm_service=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.put("/api/providers/bedrock/api-key", json={"api_key": "sk-test"})
        assert response.status_code == 400
        assert "Unknown provider" in response.json()["detail"]
        llm.store_provider_key.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_set_api_key_rejects_empty_key(self):
        app = _make_app(llm_service=_llm_with_key_store())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.put("/api/providers/anthropic/api-key", json={"api_key": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_set_api_key_503_when_the_store_is_unavailable(self):
        from assistant_runtime.services.llm.exceptions import ProviderKeyStoreUnavailableError

        llm = _llm_with_key_store(
            store_error=ProviderKeyStoreUnavailableError("Database not reachable")
        )
        app = _make_app(llm_service=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.put("/api/providers/anthropic/api-key", json={"api_key": "sk-x"})
        assert response.status_code == 503
        assert "Database not reachable" in response.json()["detail"]


class TestDeleteApiKey:
    @pytest.mark.asyncio
    async def test_delete_api_key_reports_whether_one_was_stored(self):
        for deleted in (True, False):
            llm = _llm_with_key_store(deleted=deleted)
            app = _make_app(llm_service=llm)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.delete("/api/providers/anthropic/api-key")
            assert response.status_code == 200
            assert response.json() == {
                "provider": "anthropic",
                "status": "removed",
                "was_stored": deleted,
            }
            llm.delete_provider_key.assert_awaited_once_with("anthropic")

    @pytest.mark.asyncio
    async def test_delete_api_key_rejects_unknown_provider(self):
        app = _make_app(llm_service=_llm_with_key_store())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.delete("/api/providers/bedrock/api-key")
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_delete_api_key_503_when_the_store_is_unavailable(self):
        from assistant_runtime.services.llm.exceptions import ProviderKeyStoreUnavailableError

        llm = _llm_with_key_store(
            store_error=ProviderKeyStoreUnavailableError("Encryption not configured")
        )
        app = _make_app(llm_service=llm)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.delete("/api/providers/anthropic/api-key")
        assert response.status_code == 503
