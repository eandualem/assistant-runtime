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
    async def test_defaults_use_supplied_app_settings(self):
        from assistant_runtime.app.assistant.config import AssistantConfig
        from assistant_runtime.app.settings import RuntimeSettings
        from assistant_runtime.config import AppSettings
        from assistant_runtime.main import create_app
        from assistant_runtime.services.history.config import HistoryConfig
        from assistant_runtime.services.llm.config import LLMConfig
        from assistant_runtime.services.media.config import MediaConfig

        settings = AppSettings(
            _env_file=None,
            assistant=AssistantConfig(thinking_budget=7777),
            llm=LLMConfig(
                primary_model="openai:gpt-5.6-terra",
                summarization_model="openai:gpt-5.6-luna",
            ),
            history=HistoryConfig(working_memory_model="openai:gpt-5.6-luna"),
            media=MediaConfig(
                default_image_model="google:custom-image-model",
                default_video_model="luma:custom-video-model",
            ),
        )
        app = create_app(settings=settings)
        app.state.runtime_settings = RuntimeSettings(frozen_config=settings.assistant)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            defaults = (await c.get("/api/models")).json()["defaults"]
            await app.state.runtime_settings.update(default_image_model="openai:runtime-image")
            overridden = (await c.get("/api/models")).json()["defaults"]

        assert defaults["primary_model"] == settings.llm.primary_model
        assert defaults["summarization_model"] == settings.llm.summarization_model
        assert defaults["thinking_budget"] == 7777
        assert defaults["working_memory_model"] == settings.history.working_memory_model
        assert defaults["default_image_model"] == settings.media.default_image_model
        assert defaults["default_video_model"] == settings.media.default_video_model
        assert overridden["default_image_model"] == "openai:runtime-image"
        assert overridden["thinking_budget"] == 7777

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
            summarization_model="anthropic:claude-haiku-4-5",
            thinking_budget=2000,
            subagent_thinking_budget=5000,
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            defaults = (await c.get("/api/models")).json()["defaults"]

        assert defaults["primary_model"] == "openai:gpt-5.6-terra"
        assert defaults["summarization_model"] == "anthropic:claude-haiku-4-5"
        assert defaults["thinking_budget"] == 2000
        assert defaults["subagent_thinking_budget"] == 5000
        assert defaults["subagent_model"] is None

    @pytest.mark.asyncio
    async def test_defaults_fall_back_to_frozen_settings_without_services(self, client):
        from assistant_runtime.config import AppSettings

        settings = AppSettings()
        defaults = (await client.get("/api/models")).json()["defaults"]
        assert defaults["primary_model"] == settings.llm.primary_model
        assert defaults["summarization_model"] == settings.llm.summarization_model
        assert defaults["thinking_budget"] == settings.assistant.thinking_budget
        assert defaults["default_image_model"] == settings.media.default_image_model
        assert defaults["default_video_model"] == settings.media.default_video_model
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
