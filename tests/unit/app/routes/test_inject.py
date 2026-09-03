"""Tests for /assistant/inject and /assistant/sessions endpoints."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.inject import router
from assistant_runtime.services.database.models import InboxItemORM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inbox_row(
    item_id: str = "inbox-1",
    from_agent: str = "backbone",
    message: str = "Test inject",
    severity: str = "info",
    context: dict | None = None,
    surfaced: bool = False,
    created_at: datetime | None = None,
) -> MagicMock:
    """Create a mock InboxItemORM row."""
    row = MagicMock(spec=InboxItemORM)
    row.id = item_id
    row.from_agent = from_agent
    row.message = message
    row.severity = severity
    row.context = context
    row.surfaced = surfaced
    row.created_at = created_at or datetime(2026, 2, 24, 12, 0, 0, tzinfo=UTC)
    return row


def _make_mock_db() -> MagicMock:
    """Create a mock DatabaseService with session_context."""
    mock_db = MagicMock()
    mock_db._healthy = True
    mock_session = AsyncMock()

    @asynccontextmanager
    async def fake_session_context():
        yield mock_session

    mock_db.session_context = fake_session_context
    mock_db._mock_session = mock_session
    return mock_db


def _make_app(
    db_service=None,
    assistant_service=None,
) -> FastAPI:
    """Create a minimal FastAPI app with inject routes."""
    app = FastAPI()
    app.include_router(router)

    if db_service is not None:
        app.state.database_service = db_service
    if assistant_service is not None:
        app.state.assistant_service = assistant_service

    return app


def _make_mock_assistant(session_store=None) -> MagicMock:
    """Create a mock AssistantService."""
    service = MagicMock()
    service.get_session_store.return_value = session_store
    return service


def _make_mock_session_store(*, sessions: dict | None = None) -> MagicMock:
    """Create a mock SessionStore.

    sessions: mapping of session_id -> context dict (or None for "not found").
    """
    store = MagicMock()

    async def _get_context(session_id):
        if sessions and session_id in sessions:
            return sessions[session_id]
        return None

    async def _get_session_id_for_telegram_chat(chat_id):
        if not sessions:
            return None
        for session_id, ctx in sessions.items():
            if ctx and ctx.get("telegram_chat_id") == chat_id:
                return session_id
        return None

    store.get_context_if_exists_async = AsyncMock(side_effect=_get_context)
    store.get_session_id_for_telegram_chat_async = AsyncMock(
        side_effect=_get_session_id_for_telegram_chat
    )
    return store


# ---------------------------------------------------------------------------
# POST /assistant/inject
# ---------------------------------------------------------------------------


class TestInjectMessage:
    @pytest.mark.asyncio
    async def test_inject_with_valid_session_returns_delivered(self):
        """Inject with a known session_id stores to inbox and returns 'delivered'."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()
        session_store = _make_mock_session_store(sessions={"sess-1": {"turn_number": 3}})
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(db_service=mock_db, assistant_service=assistant)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "backbone",
                        "via": "backbone",
                        "message": "Hello from Leo",
                        "sessionId": "sess-1",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "delivered"
        assert data["inbox_id"] == "inbox-1"
        assert data["session_id"] == "sess-1"

        # Verify context passed to repo.create includes session_id
        call_kwargs = mock_repo_inst.create.call_args.kwargs
        assert call_kwargs["context"]["session_id"] == "sess-1"
        assert call_kwargs["context"]["via"] == "backbone"
        assert call_kwargs["context"]["injected"] is True

    @pytest.mark.asyncio
    async def test_inject_without_session_id_returns_deferred(self):
        """Inject without sessionId stores to inbox and returns 'deferred'."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()
        assistant = _make_mock_assistant()

        app = _make_app(db_service=mock_db, assistant_service=assistant)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "backbone",
                        "via": "backbone",
                        "message": "Hello",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "deferred"
        assert data["inbox_id"] == "inbox-1"

        # Context should NOT have session_id
        call_kwargs = mock_repo_inst.create.call_args.kwargs
        assert "session_id" not in call_kwargs["context"]

    @pytest.mark.asyncio
    async def test_inject_with_nonexistent_session_returns_deferred(self):
        """Inject with unknown sessionId returns 'deferred'."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()
        session_store = _make_mock_session_store(sessions={})
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(db_service=mock_db, assistant_service=assistant)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "backbone",
                        "via": "backbone",
                        "message": "Hello",
                        "sessionId": "nonexistent-session",
                    },
                )

        assert response.status_code == 201
        assert response.json()["status"] == "deferred"

    @pytest.mark.asyncio
    async def test_inject_telegram_reply_resolves_session_from_chat_binding(self):
        """Telegram replies resolve the target assistant session from the bound chat id."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()
        session_store = _make_mock_session_store(
            sessions={"sess-assistant": {"turn_number": 3, "telegram_chat_id": "123456789"}}
        )
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(db_service=mock_db, assistant_service=assistant)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "operator",
                        "via": "telegram",
                        "message": "Reply from Telegram",
                        "telegramChatId": "123456789",
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "delivered"
        assert data["session_id"] == "sess-assistant"

        call_kwargs = mock_repo_inst.create.call_args.kwargs
        assert call_kwargs["context"]["session_id"] == "sess-assistant"
        assert call_kwargs["context"]["telegram_chat_id"] == "123456789"

    @pytest.mark.asyncio
    async def test_inject_telegram_reply_without_binding_returns_deferred(self):
        """Telegram replies without a known chat binding stay deferred."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()
        session_store = _make_mock_session_store(sessions={})
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(db_service=mock_db, assistant_service=assistant)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "operator",
                        "via": "telegram",
                        "message": "Reply from Telegram",
                        "telegramChatId": "123456789",
                    },
                )

        assert response.status_code == 201
        assert response.json()["status"] == "deferred"

    @pytest.mark.asyncio
    async def test_inject_missing_required_fields_returns_422(self):
        """Missing 'from' or 'message' returns 422."""
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Missing 'message'
            response = await c.post(
                "/assistant/inject",
                json={"from": "backbone", "via": "backbone"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_inject_missing_via_returns_422(self):
        """Missing 'via' field returns 422."""
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject",
                json={"from": "backbone", "message": "Hello"},
            )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_inject_no_db_returns_503(self):
        """When database is not available, returns 503."""
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/assistant/inject",
                json={
                    "from": "backbone",
                    "via": "backbone",
                    "message": "Hello",
                },
            )
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_inject_without_assistant_service_defers(self):
        """When assistant service isn't set, session check is skipped — deferred."""
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()

        # No assistant_service on app.state
        app = _make_app(db_service=mock_db, assistant_service=None)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app._injector.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/assistant/inject",
                    json={
                        "from": "backbone",
                        "via": "backbone",
                        "message": "Hello",
                        "sessionId": "sess-1",
                    },
                )

        assert response.status_code == 201
        assert response.json()["status"] == "deferred"


# ---------------------------------------------------------------------------
# GET /assistant/sessions
# ---------------------------------------------------------------------------


class TestListSessionsBackbone:
    @pytest.mark.asyncio
    async def test_returns_backbone_format(self):
        """Sessions are returned in backbone-expected wrapper format."""
        mock_sessions = [
            {"session_id": "sess-1", "title": "Chat 1", "turn_number": 3},
            {"session_id": "sess-2", "title": None, "turn_number": 0},
        ]
        session_store = MagicMock()
        session_store.list_sessions = AsyncMock(return_value=mock_sessions)
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(assistant_service=assistant)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")

        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert len(data["sessions"]) == 2

        s1 = data["sessions"][0]
        assert s1["id"] == "sess-1"
        assert s1["active"] is True
        assert s1["title"] == "Chat 1"
        assert s1["turn_number"] == 3

        s2 = data["sessions"][1]
        assert s2["id"] == "sess-2"
        assert s2["active"] is True

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_sessions(self):
        """When no sessions exist, returns empty sessions list."""
        session_store = MagicMock()
        session_store.list_sessions = AsyncMock(return_value=[])
        assistant = _make_mock_assistant(session_store=session_store)

        app = _make_app(assistant_service=assistant)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")

        assert response.status_code == 200
        assert response.json() == {"sessions": []}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_assistant_service(self):
        """When assistant service is not available, returns empty sessions."""
        app = _make_app(assistant_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")

        assert response.status_code == 200
        assert response.json() == {"sessions": []}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_session_store(self):
        """When session store is None, returns empty sessions."""
        assistant = _make_mock_assistant(session_store=None)
        app = _make_app(assistant_service=assistant)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/assistant/sessions")

        assert response.status_code == 200
        assert response.json() == {"sessions": []}
