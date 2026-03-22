from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant.models import AssistantResult
from lovely_assistant.main import create_app


def _create_test_app(*, assistant_service: Any = None) -> Any:
    from lovely_assistant.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()
    app.state.assistant_service = assistant_service or MagicMock()
    return app


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_chat_uses_unified_request_contract(self) -> None:
        service = AsyncMock()
        service.process_message.return_value = AssistantResult(
            content="Hello!",
            model="openai:gpt-5.4",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(assistant_service=service)

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
        request = service.process_message.await_args.args[0]
        assert request.id == "user-1"
        assert request.content == "Hi"

    @pytest.mark.asyncio
    async def test_chat_normalizes_camel_case_payload(self) -> None:
        service = AsyncMock()
        service.process_message.return_value = AssistantResult(
            content="OK",
            model="openai:gpt-5.4",
            session_id="sess-1",
            turn_number=1,
        )
        app = _create_test_app(assistant_service=service)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/chat",
                json={
                    "id": "user-1",
                    "sessionId": "sess-1",
                    "parentId": None,
                    "content": "Check agents",
                    "machineState": {"activePage": {"name": "agents", "data": {"entities": []}}},
                },
            )

        assert response.status_code == 200
        request = service.process_message.await_args.args[0]
        assert request.machine_state == {
            "active_page": {"name": "agents", "data": {"sessions": []}}
        }

    @pytest.mark.asyncio
    async def test_chat_invalid_body_returns_422(self) -> None:
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/chat", json={"session_id": "sess-1", "content": "Hi"})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_chat_rejects_steering_shape(self) -> None:
        service = AsyncMock()
        app = _create_test_app(assistant_service=service)

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
        service.process_message.assert_not_called()
