"""Tests for GET /models route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.models import router


def _make_app() -> FastAPI:
    """Create a minimal FastAPI app with models routes."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
async def client():
    """Async test client for models routes."""
    app = _make_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestEffectiveDefaults:
    @pytest.mark.asyncio
    async def test_defaults_reflect_runtime_overrides_and_the_llm_service(self):
        from unittest.mock import MagicMock

        from assistant_runtime.app.assistant.config import AssistantConfig
        from assistant_runtime.app.settings import RuntimeSettings

        app = _make_app()
        llm = MagicMock()
        llm.effective_primary_model.return_value = "openai:gpt-5.6-terra"
        llm.effective_summarization_model.return_value = "openai:gpt-5.6-luna"
        app.state.llm_service = llm
        app.state.runtime_settings = RuntimeSettings(frozen_config=AssistantConfig())
        await app.state.runtime_settings.update(
            summarization_model="anthropic:claude-haiku-4-5", subagent_thinking_budget=5000
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            defaults = (await c.get("/api/models")).json()["defaults"]

        assert defaults["primary_model"] == "openai:gpt-5.6-terra"
        assert defaults["summarization_model"] == "anthropic:claude-haiku-4-5"
        assert defaults["subagent_thinking_budget"] == 5000
        assert defaults["subagent_model"] is None

    @pytest.mark.asyncio
    async def test_defaults_fall_back_to_frozen_settings_without_services(self, client):
        defaults = (await client.get("/api/models")).json()["defaults"]
        assert isinstance(defaults["primary_model"], str)
        assert isinstance(defaults["default_image_model"], str)
        assert defaults["subagent_model"] is None


class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        response = await client.get("/api/models")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_shape(self, client):
        response = await client.get("/api/models")
        data = response.json()
        assert "models" in data
        assert "providers" in data
        assert "defaults" in data

    @pytest.mark.asyncio
    async def test_models_is_list(self, client):
        response = await client.get("/api/models")
        data = response.json()
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    @pytest.mark.asyncio
    async def test_providers_is_dict(self, client):
        response = await client.get("/api/models")
        data = response.json()
        assert isinstance(data["providers"], dict)
        assert "anthropic" in data["providers"]

    @pytest.mark.asyncio
    async def test_defaults_has_expected_keys(self, client):
        response = await client.get("/api/models")
        data = response.json()
        assert "primary_model" in data["defaults"]
        assert "summarization_model" in data["defaults"]

    @pytest.mark.asyncio
    async def test_model_entries_have_required_fields(self, client):
        response = await client.get("/api/models")
        data = response.json()
        for model in data["models"]:
            assert "id" in model
            assert "provider" in model
            assert "name" in model
            assert "capability" in model
            assert "capabilities" in model
            assert "description" in model

    @pytest.mark.asyncio
    async def test_provider_entries_have_configured(self, client):
        response = await client.get("/api/models")
        data = response.json()
        for provider in data["providers"].values():
            assert "configured" in provider
            assert isinstance(provider["configured"], bool)

    @pytest.mark.asyncio
    async def test_filter_by_capability(self, client):
        response = await client.get("/api/models", params={"capability": "text"})
        data = response.json()
        assert response.status_code == 200
        assert len(data["models"]) > 0
        for model in data["models"]:
            assert "text" in model["capabilities"]

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, client):
        response = await client.get("/api/models", params={"provider": "anthropic"})
        data = response.json()
        assert response.status_code == 200
        assert len(data["models"]) > 0
        for model in data["models"]:
            assert model["provider"] == "anthropic"

    @pytest.mark.asyncio
    async def test_combined_filter(self, client):
        response = await client.get(
            "/api/models", params={"capability": "text", "provider": "anthropic"}
        )
        data = response.json()
        assert response.status_code == 200
        assert len(data["models"]) > 0
        for model in data["models"]:
            assert model["provider"] == "anthropic"
            assert "text" in model["capabilities"]

    @pytest.mark.asyncio
    async def test_no_match_returns_empty_list(self, client):
        response = await client.get("/api/models", params={"provider": "nonexistent"})
        data = response.json()
        assert response.status_code == 200
        assert data["models"] == []

    @pytest.mark.asyncio
    async def test_invalid_capability_returns_200_empty(self, client):
        response = await client.get("/api/models", params={"capability": "teleportation"})
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
