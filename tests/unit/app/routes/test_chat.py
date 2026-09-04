from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.assistant.exceptions import AgentRunError, SessionError
from assistant_runtime.app.assistant.models import AssistantResult
from assistant_runtime.main import create_app


def _create_test_app(*, streaming_service: Any = None) -> Any:
    from assistant_runtime.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()
    app.state.streaming_service = streaming_service or MagicMock()
    return app


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_uses_unified_request_contract(self) -> None:
        service = AsyncMock()
        service.run_message.return_value = AssistantResult(
            content="Hello!",
            model="openai:gpt-5.4",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(streaming_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "user-1",
                    "session_id": "sess-1",
                    "parent_id": None,
                    "content": "Hi",
                },
            )

        assert response.status_code == 200
        assert response.json()["content"] == "Hello!"
        request = service.run_message.await_args.args[0]
        assert request.id == "user-1"
        assert request.content == "Hi"

    @pytest.mark.asyncio
    async def test_chat_normalizes_camel_case_payload(self) -> None:
        service = AsyncMock()
        service.run_message.return_value = AssistantResult(
            content="OK",
            model="openai:gpt-5.4",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(streaming_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "user-1",
                    "sessionId": "sess-1",
                    "parentId": None,
                    "content": "Check agents",
                    "hostContext": {"page": {"name": "agents", "data": {"entities": []}}},
                },
            )

        assert response.status_code == 200
        request = service.run_message.await_args.args[0]
        assert request.host_context == {"page": {"name": "agents", "data": {"entities": []}}}

    @pytest.mark.asyncio
    async def test_chat_session_rejection_is_a_409(self) -> None:
        service = AsyncMock()
        service.run_message.side_effect = SessionError("Parent message 'missing' not found")
        app = _create_test_app(streaming_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "user-2",
                    "session_id": "sess-1",
                    "parent_id": "missing",
                    "content": "Hi",
                },
            )

        assert response.status_code == 409
        assert response.json()["type"] == "SessionError"

    @pytest.mark.asyncio
    async def test_chat_run_failure_is_a_500(self) -> None:
        service = AsyncMock()
        service.run_message.side_effect = AgentRunError("boom")
        app = _create_test_app(streaming_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat", json={"id": "user-1", "session_id": "sess-1", "content": "Hi"}
            )

        assert response.status_code == 500
        assert response.json()["type"] == "AgentRunError"

    @pytest.mark.asyncio
    async def test_chat_invalid_body_returns_422(self) -> None:
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat", json={"session_id": "sess-1", "content": "Hi"}
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_rejects_steering_shape(self) -> None:
        service = AsyncMock()
        app = _create_test_app(streaming_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "steering-1",
                    "session_id": "sess-1",
                    "content": "Focus on Leo",
                    "message_type": "steering",
                },
            )

        assert response.status_code == 422
        service.run_message.assert_not_called()
