"""Tests for chat route endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant.models import AssistantResult
from lovely_assistant.main import create_app
from lovely_assistant.services.tools.models import DeferredToolRequest


def _create_test_app(
    *,
    assistant_service: Any = None,
    streaming_service: Any = None,
) -> Any:
    """Create a test app with mocked services on app.state."""
    from lovely_assistant.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()
    app.state.assistant_service = assistant_service or MagicMock()
    app.state.streaming_service = streaming_service or MagicMock()
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
    async def test_chat_deferred_tool_call(self):
        mock_service = AsyncMock()
        mock_service.process_message.return_value = AssistantResult(
            content=None,
            model="test-model",
            deferred_tool_request=DeferredToolRequest(
                request_id="req-1",
                tool_name="navigate",
                arguments={"message": "done"},
            ),
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={"session_id": "sess-1", "message": "Do something"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] is None
        assert data["deferred_tool_request"]["tool_name"] == "navigate"

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

    @pytest.mark.asyncio
    async def test_chat_continuation(self):
        mock_service = AsyncMock()
        mock_service.process_message.return_value = AssistantResult(
            content="Continued",
            model="test-model",
            session_id="sess-1",
            turn_number=2,
        )
        app = _create_test_app(assistant_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "session_id": "sess-1",
                    "message": "Continue",
                    "tool_call_id": "call-1",
                    "tool_result": "Tool done",
                },
            )

        assert response.status_code == 200
        assert response.json()["content"] == "Continued"


class TestChatStreamEndpoint:
    @pytest.mark.asyncio
    async def test_stream_returns_sse(self):
        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "text_delta", "content": "Hello"}
            yield {"type": "final_response", "content": "Hello", "model": "m", "streamed": True}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        app = _create_test_app(streaming_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={"session_id": "sess-1", "message": "Hi"},
            )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        # Parse SSE lines
        lines = response.text.strip().split("\n\n")
        assert len(lines) == 4

        import json

        events = [json.loads(line.removeprefix("data: ")) for line in lines]
        assert events[0]["type"] == "agent_status"
        assert events[0]["status"] == "started"
        assert events[1]["type"] == "text_delta"
        assert events[1]["content"] == "Hello"
        assert events[2]["type"] == "final_response"
        assert events[3]["type"] == "agent_status"
        assert events[3]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        mock_service = MagicMock()

        async def mock_stream(request):
            yield {"type": "agent_status", "status": "started"}
            yield {"type": "agent_status", "status": "completed"}

        mock_service.stream_message = mock_stream
        app = _create_test_app(streaming_service=mock_service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat/stream",
                json={"session_id": "sess-1", "message": "Hi"},
            )

        assert response.status_code == 200
        lines = response.text.strip().split("\n\n")
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_stream_invalid_body(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat/stream", json={"bad": "data"})

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
