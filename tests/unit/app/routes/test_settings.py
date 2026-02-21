"""Tests for GET/PATCH /settings routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.routes.settings import get_runtime_settings, router
from lovely_assistant.app.settings import RuntimeSettings


def _make_app(runtime_settings: RuntimeSettings | None = None) -> FastAPI:
    """Create a minimal FastAPI app with settings routes."""
    app = FastAPI()
    app.include_router(router, prefix="/api")

    rs = runtime_settings or RuntimeSettings(frozen_config=AssistantConfig())

    app.dependency_overrides[get_runtime_settings] = lambda: rs
    return app


@pytest.fixture
async def client():
    """Async test client for settings routes."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestGetSettings:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        response = await client.get("/api/settings")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_shape(self, client):
        response = await client.get("/api/settings")
        data = response.json()
        assert "values" in data
        assert "updated_at" in data
        assert "default_model" in data["values"]
        assert "temperature" in data["values"]
        assert "thinking_budget" in data["values"]
        assert "max_turns" in data["values"]
        assert "enable_working_memory" in data["values"]

    @pytest.mark.asyncio
    async def test_response_includes_new_fields(self, client):
        response = await client.get("/api/settings")
        data = response.json()
        assert "summarization_model" in data["values"]
        assert "working_memory_model" in data["values"]
        assert "default_image_model" in data["values"]
        assert "default_video_model" in data["values"]
        assert "subagent_model" in data["values"]
        assert "subagent_thinking_budget" in data["values"]

    @pytest.mark.asyncio
    async def test_default_sources(self, client):
        response = await client.get("/api/settings")
        data = response.json()
        for field_info in data["values"].values():
            assert field_info["source"] == "config_default"


class TestPatchSettings:
    @pytest.mark.asyncio
    async def test_update_settings(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"temperature": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["values"]["temperature"]["value"] == 0.5
        assert data["values"]["temperature"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_null_clears_override(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            await c.patch("/api/settings", json={"temperature": 0.5})
            response = await c.patch("/api/settings", json={"temperature": None})
        data = response.json()
        assert data["values"]["temperature"]["source"] == "config_default"

    @pytest.mark.asyncio
    async def test_unknown_field_422(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"nonexistent": 42})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_out_of_range_422(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"temperature": 5.0})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_returns_updated_state(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch(
                "/api/settings",
                json={"temperature": 0.7, "max_turns": 20},
            )
        data = response.json()
        assert data["values"]["temperature"]["value"] == 0.7
        assert data["values"]["max_turns"]["value"] == 20
        assert data["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_patch_includes_persisted_false_without_db(self):
        """PATCH without DB returns persisted=false."""
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"temperature": 0.5})
        data = response.json()
        assert "persisted" in data
        assert data["persisted"] is False

    @pytest.mark.asyncio
    async def test_patch_includes_persisted_true_with_db(self):
        """PATCH with working DB returns persisted=true."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_db = MagicMock()
        mock_db._healthy = True
        mock_session = AsyncMock()

        @asynccontextmanager
        async def fake_session_context():
            yield mock_session

        mock_db.session_context = fake_session_context

        mock_repo = MagicMock()
        mock_repo.save = AsyncMock()
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)
        app = _make_app(rs)

        with patch(
            "lovely_assistant.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.patch("/api/settings", json={"temperature": 0.5})

        data = response.json()
        assert "persisted" in data
        assert data["persisted"] is True

    @pytest.mark.asyncio
    async def test_update_summarization_model(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch(
                "/api/settings", json={"summarization_model": "openai:gpt-4o-mini"}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["values"]["summarization_model"]["value"] == "openai:gpt-4o-mini"
        assert data["values"]["summarization_model"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_update_subagent_thinking_budget(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        app = _make_app(rs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"subagent_thinking_budget": 5000})
        assert response.status_code == 200
        data = response.json()
        assert data["values"]["subagent_thinking_budget"]["value"] == 5000
        assert data["values"]["subagent_thinking_budget"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_subagent_thinking_budget_out_of_range_422(self):
        app = _make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/api/settings", json={"subagent_thinking_budget": 200000})
        assert response.status_code == 422
