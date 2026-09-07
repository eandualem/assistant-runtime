"""Integration tests for HTTP routes — full request chain with real app."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.main import create_app
from tests.integration.conftest import make_artifact_service

from .conftest import _make_mock_agent


def _payload(
    *, message_id: str, session_id: str, parent_id: str | None, content: str
) -> dict[str, object]:
    return {
        "id": message_id,
        "session_id": session_id,
        "parent_id": parent_id,
        "content": content,
    }


@pytest.fixture
async def integration_client(monkeypatch):
    """Create a real app with services manually wired (ASGITransport skips lifespan)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-routes")

    from assistant_runtime.app.assistant.interface import AssistantService
    from assistant_runtime.app.settings import RuntimeSettings
    from assistant_runtime.app.streaming.interface import StreamingService
    from assistant_runtime.base.lifecycle import LifecycleManager
    from assistant_runtime.config import AppSettings
    from assistant_runtime.services.history.interface import HistoryService
    from assistant_runtime.services.llm.interface import LlmService
    from assistant_runtime.services.tools.interface import ToolService

    app = create_app()
    settings = AppSettings()

    # Wire services manually (ASGITransport doesn't trigger lifespan)
    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle

    runtime_settings = RuntimeSettings(frozen_config=settings.assistant)
    app.state.runtime_settings = runtime_settings

    llm_service = LlmService(config=settings.llm)
    history_service = HistoryService(config=settings.history, llm_service=llm_service)
    artifact_service = make_artifact_service()

    tool_service = ToolService(config=settings.tools, artifact_service=artifact_service)
    assistant_service = AssistantService(
        config=settings.assistant,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        artifact_service=artifact_service,
        runtime_settings=runtime_settings,
    )

    await lifecycle.register("llm_service", llm_service)
    await lifecycle.register("history_service", history_service)

    await lifecycle.register("artifact_service", artifact_service)
    await lifecycle.register("tool_service", tool_service)
    await lifecycle.register("assistant_service", assistant_service)
    await lifecycle.start_all()

    app.state.llm_service = llm_service
    app.state.history_service = history_service
    app.state.tool_service = tool_service
    app.state.assistant_service = assistant_service

    # Streaming service accesses sessions via assistant_service
    streaming_service = StreamingService(
        config=settings.streaming,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=assistant_service,
    )
    await streaming_service.start()
    app.state.streaming_service = streaming_service

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, app

    await lifecycle.stop_all()


class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, integration_client):
        """Health endpoint works with fully wired app."""
        client, app = integration_client
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert "components" in data


class TestChatRouteIntegration:
    @pytest.mark.asyncio
    async def test_chat_full_chain(self, integration_client):
        """POST /api/chat flows through all real services."""
        client, app = integration_client
        mock_agent = _make_mock_agent("Integration response")

        with patch(
            "assistant_runtime.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            response = await client.post(
                "/api/chat",
                json=_payload(
                    message_id="user-1",
                    session_id="route-integ",
                    parent_id=None,
                    content="Hello",
                ),
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Integration response"
        assert data["session_id"] == "route-integ"
        assert data["turn_number"] == 1

    @pytest.mark.asyncio
    async def test_chat_validation_error(self, integration_client):
        """Invalid request body returns 422."""
        client, app = integration_client
        response = await client.post("/api/chat", json={"invalid": "body"})
        assert response.status_code == 422


class TestSessionRouteIntegration:
    @pytest.mark.asyncio
    async def test_chat_then_get_session(self, integration_client):
        """After a chat, the session endpoint returns session info."""
        client, app = integration_client
        mock_agent = _make_mock_agent("First message")

        with patch(
            "assistant_runtime.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            await client.post(
                "/api/chat",
                json=_payload(
                    message_id="user-1",
                    session_id="sess-route",
                    parent_id=None,
                    content="Hi",
                ),
            )

        response = await client.get("/api/sessions/sess-route")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-route"
        assert data["turn_number"] == 1

    @pytest.mark.asyncio
    async def test_chat_then_delete_session(self, integration_client):
        """After a chat, deleting the session clears it."""
        client, app = integration_client
        mock_agent = _make_mock_agent("To be deleted")

        with patch(
            "assistant_runtime.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            await client.post(
                "/api/chat",
                json=_payload(
                    message_id="user-1",
                    session_id="sess-delete",
                    parent_id=None,
                    content="Hi",
                ),
            )

        # Delete
        response = await client.delete("/api/sessions/sess-delete")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

        # Get should now 404
        response = await client.get("/api/sessions/sess-delete")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self, integration_client):
        """GET for a non-existent session returns 404."""
        client, app = integration_client
        response = await client.get("/api/sessions/no-such-session")
        assert response.status_code == 404


class TestModelsRouteIntegration:
    @pytest.mark.asyncio
    async def test_get_models_route_exists(self, integration_client):
        """GET /api/models returns 200 with expected shape."""
        client, app = integration_client
        response = await client.get("/api/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "providers" in data
        assert "defaults" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0


class TestSettingsRouteIntegration:
    @pytest.mark.asyncio
    async def test_get_settings_route_exists(self, integration_client):
        """GET /api/settings returns 200 with expected shape."""
        client, app = integration_client
        response = await client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "values" in data
        assert "updated_at" in data

    @pytest.mark.asyncio
    async def test_patch_settings_route_exists(self, integration_client):
        """PATCH /api/settings updates and returns settings."""
        client, app = integration_client
        response = await client.patch("/api/settings", json={"temperature": 0.5})
        assert response.status_code == 200
        data = response.json()
        assert data["values"]["temperature"]["value"] == 0.5
        assert data["values"]["temperature"]["source"] == "runtime"
