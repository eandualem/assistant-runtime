"""Tests for /inbox routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.inbox import router
from assistant_runtime.services.database.models import InboxItemORM

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_inbox_row(
    item_id: str = "item-1",
    from_agent: str = "leo",
    message: str = "Hello from Leo",
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
    row.created_at = created_at or datetime(2026, 2, 21, 12, 0, 0, tzinfo=UTC)
    return row


def _make_app(db_service=None) -> FastAPI:
    """Create a minimal FastAPI app with inbox routes."""
    app = FastAPI()
    app.include_router(router)

    if db_service is not None:
        app.state.database_service = db_service

    return app


def _make_mock_db(repo_mock: MagicMock | None = None) -> MagicMock:
    """Create a mock DatabaseService with session_context."""
    mock_db = MagicMock()
    mock_db._healthy = True
    mock_session = AsyncMock()

    @asynccontextmanager
    async def fake_session_context():
        yield mock_session

    mock_db.session_context = fake_session_context
    mock_db._mock_session = mock_session  # expose for test setup
    return mock_db


# ---------------------------------------------------------------------------
# POST /inbox
# ---------------------------------------------------------------------------


class TestPostInbox:
    @pytest.mark.asyncio
    async def test_creates_item_returns_201(self):
        mock_row = _make_inbox_row()
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/inbox",
                    json={"from": "leo", "message": "Hello from Leo"},
                )

        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "item-1"
        assert data["from_agent"] == "leo"
        assert data["message"] == "Hello from Leo"
        assert data["severity"] == "info"
        assert data["surfaced"] is False

    @pytest.mark.asyncio
    async def test_validates_required_fields(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Missing 'message'
            response = await c.post("/inbox", json={"from": "leo"})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_validates_empty_from(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/inbox", json={"from": "", "message": "Hello"})

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_no_db_returns_503(self):
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/inbox",
                json={"from": "leo", "message": "Hello"},
            )

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_accepts_severity_and_context(self):
        mock_row = _make_inbox_row(severity="urgent", context={"issue": 42})
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.create = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/inbox",
                    json={
                        "from": "leo",
                        "message": "Urgent!",
                        "severity": "urgent",
                        "context": {"issue": 42},
                    },
                )

        assert response.status_code == 201
        data = response.json()
        assert data["severity"] == "urgent"
        assert data["context"] == {"issue": 42}


# ---------------------------------------------------------------------------
# GET /inbox
# ---------------------------------------------------------------------------


class TestGetInbox:
    @pytest.mark.asyncio
    async def test_returns_unsurfaced_items(self):
        rows = [_make_inbox_row(item_id="item-1"), _make_inbox_row(item_id="item-2")]
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.list_unsurfaced = AsyncMock(return_value=rows)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/inbox?surfaced=false")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["id"] == "item-1"

    @pytest.mark.asyncio
    async def test_returns_all_items_without_filter(self):
        rows = [_make_inbox_row(item_id="item-1")]
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.list_all = AsyncMock(return_value=rows)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/inbox")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

    @pytest.mark.asyncio
    async def test_no_db_returns_503(self):
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/inbox")

        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_db_error_returns_503(self):
        """DB is configured but session_context raises — should return 503."""
        mock_db = MagicMock()
        mock_db._healthy = True

        @asynccontextmanager
        async def _failing_context():
            raise RuntimeError("DB down")
            yield  # noqa: F401

        mock_db.session_context = _failing_context

        app = _make_app(db_service=mock_db)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/inbox")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# PATCH /inbox/{item_id}/surfaced
# ---------------------------------------------------------------------------


class TestPatchSurfaced:
    @pytest.mark.asyncio
    async def test_marks_item_surfaced(self):
        mock_row = _make_inbox_row(surfaced=True)
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.mark_surfaced = AsyncMock(return_value=mock_row)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.patch("/inbox/item-1/surfaced")

        assert response.status_code == 200
        data = response.json()
        assert data["surfaced"] is True

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_item(self):
        mock_db = _make_mock_db()

        app = _make_app(db_service=mock_db)
        with pytest.MonkeyPatch.context() as mp:
            mock_repo_cls = MagicMock()
            mock_repo_inst = MagicMock()
            mock_repo_inst.mark_surfaced = AsyncMock(return_value=None)
            mock_repo_cls.return_value = mock_repo_inst
            mp.setattr(
                "assistant_runtime.app.routes.inbox.InboxRepository",
                mock_repo_cls,
            )

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.patch("/inbox/nonexistent/surfaced")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_no_db_returns_503(self):
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch("/inbox/item-1/surfaced")

        assert response.status_code == 503
