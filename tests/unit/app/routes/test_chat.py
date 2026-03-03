"""Tests for chat route endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant.models import AssistantResult
from lovely_assistant.main import create_app


def _create_test_app(
    *,
    assistant_service: Any = None,
) -> Any:
    """Create a test app with mocked services on app.state."""
    from lovely_assistant.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()
    app.state.assistant_service = assistant_service or MagicMock()
    return app


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_returns_result(self):
        mock_service = AsyncMock()
        mock_service.process_message.return_value = AssistantResult(
            content="Hello!",
            model="test-model",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "sess-1", "message": "Hi"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Hello!"
        assert data["model"] == "test-model"
        assert data["session_id"] == "sess-1"
        assert data["turn_number"] == 1

    @pytest.mark.asyncio
    async def test_chat_with_machine_state(self):
        mock_service = AsyncMock()
        mock_service.process_message.return_value = AssistantResult(
            content="OK",
            model="test-model",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "session_id": "sess-1",
                    "message": "Check agents",
                    "machine_state": {"current_state": "agents"},
                },
            )

        assert response.status_code == 200
        # Verify machine_state was passed through
        call_args = mock_service.process_message.call_args[0][0]
        assert call_args.machine_state == {"current_state": "agents"}

    @pytest.mark.asyncio
    async def test_chat_invalid_body(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"bad": "data"})

        assert response.status_code == 422


class TestErrorHandlers:
    @pytest.mark.asyncio
    async def test_assistant_error_returns_500(self):
        from lovely_assistant.app.assistant.exceptions import AgentRunError

        mock_service = AsyncMock()
        mock_service.process_message.side_effect = AgentRunError("Agent failed")
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "sess-1", "message": "Hi"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Agent failed"
        assert data["type"] == "AgentRunError"

    @pytest.mark.asyncio
    async def test_session_error_returns_500(self):
        from lovely_assistant.app.assistant.exceptions import SessionError

        mock_service = AsyncMock()
        mock_service.process_message.side_effect = SessionError("Bad session")
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "sess-1", "message": "Hi"},
            )

        assert response.status_code == 500
        data = response.json()
        assert data["type"] == "SessionError"
