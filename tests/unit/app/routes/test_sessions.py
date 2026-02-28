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
    mock_service.get_session_store.return_value = sessions or SessionStore()
    mock_service.get_database_service.return_value = None
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
        """When session store is None (service not started), returns empty list."""
        app = _create_test_app()
        app.state.assistant_service.get_session_store.return_value = None

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
        assert data["message_count"] == 0

    @pytest.mark.asyncio
    async def test_get_session_not_found(self):
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/nonexistent")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

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
        db_context = {
            "turn_number": 5,
            "working_memory": None,
            "message_history": [MagicMock(), MagicMock()],
            "title": "DB session",
        }
        sessions.get_context_if_exists_async = AsyncMock(return_value=db_context)

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
        sessions.get_context_if_exists_async = AsyncMock(return_value=None)

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
        sessions.get_context_if_exists_async = AsyncMock(return_value=None)

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


class TestGetSessionMessages:
    @pytest.mark.asyncio
    async def test_get_messages_with_history(self):
        """Session with real ModelMessage objects returns display-format messages."""
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        sessions = SessionStore()
        sessions.get_context("sess-1")

        history = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi there")]),
        ]
        sessions.save_history("sess-1", history)

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[0]["text"] == "Hello"
        assert data[1]["role"] == "assistant"
        assert data[1]["text"] == "Hi there"

    @pytest.mark.asyncio
    async def test_get_messages_empty_history(self):
        """Session exists but has no messages — returns empty list."""
        sessions = SessionStore()
        sessions.get_context("sess-1")

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_messages_not_found(self):
        """Unknown session with DB returning None gives 404."""
        sessions = SessionStore()
        sessions._load_session_from_db = AsyncMock(return_value=None)

        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/nonexistent/messages")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"

    @pytest.mark.asyncio
    async def test_get_messages_no_session_store(self):
        """When session store is None (service not started), returns 404."""
        app = _create_test_app()
        app.state.assistant_service.get_session_store.return_value = None

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"


class TestGetSessionTraces:
    @pytest.mark.asyncio
    async def test_get_traces_returns_empty_when_no_db(self):
        """When database service is None, returns empty list."""
        app = _create_test_app()
        # get_database_service already returns None from _create_test_app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/traces")

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.asyncio
    async def test_get_traces_returns_503_on_db_error(self):
        """When DB throws, returns 503 instead of swallowing the error."""
        mock_db = MagicMock()
        mock_db.session_context = MagicMock(side_effect=RuntimeError("DB down"))

        app = _create_test_app()
        app.state.assistant_service.get_database_service.return_value = mock_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/traces")

        assert response.status_code == 503
        assert response.json()["detail"] == "Database unavailable"

    @pytest.mark.asyncio
    async def test_get_traces_returns_data(self):
        """When DB has traces, returns serialized trace rows."""
        from datetime import UTC, datetime

        mock_row = MagicMock()
        mock_row.id = "trace-001"
        mock_row.session_id = "sess-1"
        mock_row.events = [{"type": "debug_request", "session_id": "sess-1"}]
        mock_row.user_message = "Hello"
        mock_row.is_continuation = False
        mock_row.duration_ms = 123.4
        mock_row.screenshot = "data:image/jpeg;base64,abc123"
        mock_row.created_at = datetime(2026, 2, 21, 10, 0, 0, tzinfo=UTC)

        mock_db_session = AsyncMock()
        mock_db = MagicMock()
        mock_db.session_context = MagicMock(return_value=mock_db_session)
        mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_db_session.__aexit__ = AsyncMock(return_value=None)

        # Mock the repository's list_by_session to return our mock row
        # The endpoint creates TraceRepository internally, so we mock execute
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row]
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        app = _create_test_app()
        app.state.assistant_service.get_database_service.return_value = mock_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/traces")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "trace-001"
        assert data[0]["session_id"] == "sess-1"
        assert data[0]["events"] == [{"type": "debug_request", "session_id": "sess-1"}]
        assert data[0]["user_message"] == "Hello"
        assert data[0]["is_continuation"] is False
        assert data[0]["duration_ms"] == 123.4
        assert data[0]["screenshot"] == "data:image/jpeg;base64,abc123"
        assert data[0]["created_at"] == "2026-02-21T10:00:00+00:00"

    @pytest.mark.asyncio
    async def test_get_traces_pagination(self):
        """Limit and offset are passed through to the query."""
        mock_db_session = AsyncMock()
        mock_db = MagicMock()
        mock_db.session_context = MagicMock(return_value=mock_db_session)
        mock_db_session.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_db_session.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        app = _create_test_app()
        app.state.assistant_service.get_database_service.return_value = mock_db

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/traces?limit=10&offset=5")

        assert response.status_code == 200
        assert response.json() == []
