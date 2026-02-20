"""Tests for session management route endpoints."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

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


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_empty(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_list_sessions_returns_in_memory(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.increment_turn("sess-1")
        sessions.get_context("sess-2")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        ids = {s["session_id"] for s in data}
        assert ids == {"sess-1", "sess-2"}

    @pytest.mark.asyncio
    async def test_list_sessions_with_limit(self):
        sessions = SessionStore()
        for i in range(5):
            sessions.get_context(f"sess-{i}")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions?limit=2&offset=0")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_none_sessions(self):
        """When _sessions is None, returns empty list."""
        app = _create_test_app()
        app.state.assistant_service._sessions = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions")

        assert response.status_code == 200
        assert response.json() == []


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

    @pytest.mark.asyncio
    async def test_get_session_loads_from_db(self):
        """When session not in memory but exists in DB, loads and returns it."""
        sessions = SessionStore()
        sessions._load_session_from_db = AsyncMock(
            return_value={
                "turn_number": 5,
                "working_memory": None,
                "pending_tool_call": None,
                "message_history": [MagicMock(), MagicMock()],
                "title": "DB session",
            }
        )

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/db-sess")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "db-sess"
        assert data["turn_number"] == 5
        assert data["message_count"] == 2

    @pytest.mark.asyncio
    async def test_get_session_not_in_memory_or_db(self):
        """When session not in memory and DB returns None, 404."""
        sessions = SessionStore()
        sessions._load_session_from_db = AsyncMock(return_value=None)

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/nonexistent")

        assert response.status_code == 404


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_delete_existing_session(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions.increment_turn("sess-1")
        sessions.delete_session = AsyncMock()

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/sessions/sess-1")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-1"
        assert data["deleted"] is True

    @pytest.mark.asyncio
    async def test_delete_session_not_found(self):
        sessions = SessionStore()
        sessions._load_session_from_db = AsyncMock(return_value=None)

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/sessions/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    @pytest.mark.asyncio
    async def test_delete_then_get_returns_404(self):
        sessions = SessionStore()
        sessions.get_context("sess-1")
        sessions._load_session_from_db = AsyncMock(return_value=None)

        # Make delete_session actually remove from memory
        original_sessions_dict = sessions._sessions

        async def mock_delete(session_id):
            original_sessions_dict.pop(session_id, None)

        sessions.delete_session = AsyncMock(side_effect=mock_delete)

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Delete
            response = await client.delete("/api/sessions/sess-1")
            assert response.status_code == 200

            # Get should now 404
            response = await client.get("/api/sessions/sess-1")
            assert response.status_code == 404
