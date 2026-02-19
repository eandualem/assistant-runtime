"""Integration tests for HTTP routes — full request chain with real app."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.main import create_app

from .conftest import _make_mock_agent, _make_mock_agent_run


@pytest.fixture
async def integration_client(monkeypatch):
    """Create a real app with services manually wired (ASGITransport skips lifespan)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-routes")

    from lovely_assistant.app.assistant.interface import AssistantService
    from lovely_assistant.app.streaming.interface import StreamingService
    from lovely_assistant.base.lifecycle import LifecycleManager
    from lovely_assistant.config import AppSettings
    from lovely_assistant.services.history.interface import HistoryService
    from lovely_assistant.services.llm.interface import LlmService
    from lovely_assistant.services.tools.interface import ToolService

    app = create_app()
    settings = AppSettings()

    # Wire services manually (ASGITransport doesn't trigger lifespan)
    lifecycle = LifecycleManager()
    app.state.lifecycle = lifecycle

    llm_service = LlmService(config=settings.llm)
    history_service = HistoryService(config=settings.history, llm_service=llm_service)
    tool_service = ToolService(config=settings.tools)
    assistant_service = AssistantService(
        config=settings.assistant,
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
    )

    await lifecycle.register("llm_service", llm_service)
    await lifecycle.register("history_service", history_service)
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
        llm_service=llm_service,
        history_service=history_service,
        tool_service=tool_service,
        assistant_service=assistant_service,
        assistant_config=settings.assistant,
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
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            response = await client.post(
                "/api/chat",
                json={"session_id": "route-integ", "message": "Hello"},
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


class TestStreamRouteIntegration:
    @pytest.mark.asyncio
    async def test_stream_full_chain(self, integration_client):
        """POST /api/chat/stream returns SSE events through real services."""
        client, app = integration_client
        mock_run = _make_mock_agent_run("Streamed!")

        mock_agent = MagicMock()
        mock_agent.iter = MagicMock(return_value=mock_run)

        with patch(
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            response = await client.post(
                "/api/chat/stream",
                json={"session_id": "stream-route", "message": "Hi"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Parse SSE
        lines = [line for line in response.text.strip().split("\n\n") if line]
        events = [json.loads(line.removeprefix("data: ")) for line in lines]

        types = [e["type"] for e in events]
        assert "agent_status" in types
        assert "final_response" in types

        final = [e for e in events if e["type"] == "final_response"]
        assert final[0]["content"] == "Streamed!"

    @pytest.mark.asyncio
    async def test_stream_validation_error(self, integration_client):
        """Invalid stream request body returns 422."""
        client, app = integration_client
        response = await client.post("/api/chat/stream", json={"bad": "body"})
        assert response.status_code == 422


class TestSessionRouteIntegration:
    @pytest.mark.asyncio
    async def test_chat_then_get_session(self, integration_client):
        """After a chat, the session endpoint returns session info."""
        client, app = integration_client
        mock_agent = _make_mock_agent("First message")

        with patch(
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            await client.post(
                "/api/chat",
                json={"session_id": "sess-route", "message": "Hi"},
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
            "lovely_assistant.services.llm.interface.LlmService.build_agent",
            return_value=mock_agent,
        ):
            await client.post(
                "/api/chat",
                json={"session_id": "sess-delete", "message": "Hi"},
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
