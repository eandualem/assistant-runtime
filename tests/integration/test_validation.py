"""End-to-end validation tests — app startup, config composition, route registration, edge cases."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant.interface import AssistantService
from lovely_assistant.app.settings import RuntimeSettings
from lovely_assistant.app.streaming.interface import StreamingService
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.config import AppSettings
from lovely_assistant.main import create_app
from lovely_assistant.services.history.interface import HistoryService
from lovely_assistant.services.llm.interface import LlmService
from lovely_assistant.services.tools.interface import ToolService

from .conftest import _make_mock_agent


@pytest.fixture
async def full_app_client(monkeypatch):
    """App with all services wired — mirrors lifespan behavior."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-validation")

    settings = AppSettings()
    app = create_app()

    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle

    runtime_settings = RuntimeSettings(frozen_config=settings.assistant)
    app.state.runtime_settings = runtime_settings

    llm_service = LlmService(config=settings.llm)
    history_service = HistoryService(config=settings.history, llm_service=llm_service)
    tool_service = ToolService(config=settings.tools)
    assistant_service = AssistantService(
        config=settings.assistant,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        runtime_settings=runtime_settings,
    )

    await lifecycle.register("llm_service", llm_service)
    await lifecycle.register("history_service", history_service)
    await lifecycle.register("tool_service", tool_service)
    await lifecycle.register("assistant_service", assistant_service)
    await lifecycle.start_all()

    streaming_service = StreamingService(
        config=settings.streaming,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=assistant_service,
        runtime_settings=runtime_settings,
        assistant_config=settings.assistant,
    )
    await lifecycle.register("streaming_service", streaming_service)
    await streaming_service.start()

    app.state.llm_service = llm_service
    app.state.history_service = history_service
    app.state.tool_service = tool_service
    app.state.assistant_service = assistant_service
    app.state.streaming_service = streaming_service

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, app

    await lifecycle.stop_all()


class TestAppStartup:
    def test_create_app_returns_fastapi(self):
        """create_app() produces a valid FastAPI instance."""
        app = create_app()
        assert app.title == "Lovely Assistant"
        assert app.version == "0.1.0"

    def test_all_routes_registered(self):
        """App has all expected route paths registered."""
        app = create_app()
        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in route_paths
        assert "/api/chat" in route_paths
        assert "/api/sessions/{session_id}" in route_paths
        assert "/api/settings" in route_paths
        assert "/api/models" in route_paths

    def test_route_methods(self):
        """Routes have the correct HTTP methods."""
        app = create_app()
        routes_by_path: dict[str, set[str]] = {}
        for r in app.routes:
            if hasattr(r, "path") and hasattr(r, "methods"):
                routes_by_path.setdefault(r.path, set()).update(r.methods)

        assert "GET" in routes_by_path.get("/health", set())
        assert "POST" in routes_by_path.get("/api/chat", set())
        assert "GET" in routes_by_path.get("/api/sessions/{session_id}", set())
        assert "DELETE" in routes_by_path.get("/api/sessions/{session_id}", set())
        assert "GET" in routes_by_path.get("/api/settings", set())
        assert "PATCH" in routes_by_path.get("/api/settings", set())
        assert "GET" in routes_by_path.get("/api/models", set())

    @pytest.mark.asyncio
    async def test_cors_headers(self, full_app_client):
        """Responses include CORS headers."""
        client, app = full_app_client
        response = await client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in response.headers


class TestConfigCompositionValidation:
    def test_app_settings_schema(self):
        """AppSettings schema includes all module configs."""
        schema = AppSettings.model_json_schema()
        properties = schema.get("properties", {})
        assert "llm" in properties
        assert "history" in properties
        assert "tools" in properties
        assert "assistant" in properties
        assert "streaming" in properties

    def test_env_var_override(self, monkeypatch):
        """Nested env vars override defaults."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-env")
        monkeypatch.setenv("ASSISTANT__MAX_TURNS", "25")
        settings = AppSettings()
        assert settings.assistant.max_turns == 25

    def test_default_config_composition(self):
        """All module configs compose with valid defaults."""
        settings = AppSettings()
        assert settings.llm.primary_model is not None
        assert settings.history.token_budget > 0
        assert settings.tools.max_tools_per_request > 0
        assert settings.assistant.max_turns > 0
        assert settings.streaming.debounce_seconds >= 0


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_message(self, full_app_client):
        """Empty message body is accepted (not a validation error)."""
        client, app = full_app_client
        mock_agent = _make_mock_agent("Response to empty")

        with patch(
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            response = await client.post(
                "/api/chat",
                json={"session_id": "edge-empty", "message": ""},
            )

        assert response.status_code == 200
        assert response.json()["content"] == "Response to empty"

    @pytest.mark.asyncio
    async def test_long_session_id(self, full_app_client):
        """Very long session_id is handled without error."""
        client, app = full_app_client
        long_id = "s" * 1000
        mock_agent = _make_mock_agent("Long session OK")

        with patch(
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            response = await client.post(
                "/api/chat",
                json={"session_id": long_id, "message": "test"},
            )

        assert response.status_code == 200
        assert response.json()["session_id"] == long_id

    @pytest.mark.asyncio
    async def test_sequential_requests_same_session(self, full_app_client):
        """Multiple requests to same session accumulate turns correctly."""
        client, app = full_app_client

        for i in range(1, 4):
            mock_agent = _make_mock_agent(f"Reply {i}")
            with patch(
                "lovely_assistant.services.llm.interface.LlmService.build_agent",
                return_value=mock_agent,
            ):
                response = await client.post(
                    "/api/chat",
                    json={"session_id": "edge-multi", "message": f"msg{i}"},
                )
            assert response.status_code == 200

        # Session should have 3 turns
        sess_response = await client.get("/api/sessions/edge-multi")
        assert sess_response.status_code == 200
        assert sess_response.json()["turn_number"] == 3

    @pytest.mark.asyncio
    async def test_health_reports_all_components(self, full_app_client):
        """Health endpoint reports all registered service components."""
        client, app = full_app_client
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        component_names = set(data["components"].keys())
        assert "llm_service" in component_names
        assert "history_service" in component_names
        assert "tool_service" in component_names
        assert "assistant_service" in component_names
        assert "streaming_service" in component_names


class TestPublicApiExports:
    """Verify all modules export their public API correctly."""

    def test_llm_exports(self):
        from lovely_assistant.services.llm import LLMConfig, LLMResult, LlmService

        assert LlmService is not None
        assert LLMConfig is not None
        assert LLMResult is not None

    def test_history_exports(self):
        from lovely_assistant.services.history import (
            HistoryConfig,
            HistoryService,
            WorkingMemory,
        )

        assert HistoryService is not None
        assert HistoryConfig is not None
        assert WorkingMemory is not None

    def test_tools_exports(self):
        from lovely_assistant.services.tools import (
            ToolConfig,
            ToolDefinition,
            ToolService,
        )

        assert ToolService is not None
        assert ToolConfig is not None
        assert ToolDefinition is not None

    def test_assistant_exports(self):
        from lovely_assistant.app.assistant import (
            AssistantRequest,
            AssistantResult,
            AssistantService,
        )

        assert AssistantService is not None
        assert AssistantRequest is not None
        assert AssistantResult is not None

    def test_streaming_exports(self):
        from lovely_assistant.app.streaming import StreamingService

        assert StreamingService is not None

    def test_routes_exports(self):
        from lovely_assistant.app.routes import router

        assert router is not None

    def test_base_exports(self):
        from lovely_assistant.base import (
            LifecycleAware,
            LifecycleManager,
            LovelyAssistantError,
            instrument,
        )

        assert LifecycleManager is not None
        assert LifecycleAware is not None
        assert LovelyAssistantError is not None
        assert instrument is not None
