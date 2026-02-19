"""Tests for session management route endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.main import create_app


def _create_test_app(*, sessions: SessionStore | None = None) -> Any:
    """Create a test app with a real SessionStore on a mock AssistantService."""
    from lovely_assistant.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()

    mock_service = MagicMock()
    mock_service._sessions = sessions or SessionStore()
    app.state.assistant_service = mock_service
    app.state.streaming_service = MagicMock()
    return app


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_existing_session(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.increment_turn("sess-1")
        sessions.increment_turn("sess-1")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-1"
        assert data["turn_number"] == 2
        assert data["has_pending_tool_call"] is False
        assert data["message_count"] == 0

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    @pytest.mark.asyncio
    async def test_get_session_with_pending_tool_call(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.set_pending_tool_call("sess-1", "call-1", "ui_notify")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1")

        assert response.status_code == 200
        assert response.json()["has_pending_tool_call"] is True

    @pytest.mark.asyncio
    async def test_get_session_with_history(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.save_history("sess-1", [MagicMock(), MagicMock(), MagicMock()])

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1")

        assert response.status_code == 200
        assert response.json()["message_count"] == 3


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_existing_session(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.increment_turn("sess-1")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/sessions/sess-1")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-1"
        assert data["deleted"] is True

        # Verify session is gone
        assert not sessions.has_session("sess-1")

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/sessions/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    @pytest.mark.asyncio
    async def test_delete_then_get_returns_404(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Delete
            response = await client.delete("/api/sessions/sess-1")
            assert response.status_code == 200

            # Get should now 404
            response = await client.get("/api/sessions/sess-1")
            assert response.status_code == 404
