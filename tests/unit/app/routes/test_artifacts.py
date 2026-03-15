"""Tests for /artifacts routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.routes.artifacts import router

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 2, 22, 12, 0, 0, tzinfo=UTC)


def _make_row(
    *,
    row_id: int = 1,
    name: str = "persona",
    version: int = 1,
    is_active: bool = True,
    content: str = "test content",
    proposed_by: str = "system",
    created_at: datetime | None = None,
) -> MagicMock:
    """Create a mock ArtifactORM row."""
    row = MagicMock()
    row.id = row_id
    row.name = name
    row.content = content
    row.version = version
    row.is_active = is_active
    row.proposed_by = proposed_by
    row.created_at = created_at or _NOW
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


def _make_app(db_service=None) -> FastAPI:
    """Create a minimal FastAPI app with artifact routes."""
    app = FastAPI()
    app.include_router(router)

    if db_service is not None:
        app.state.database_service = db_service

    return app


def _patch_repo(mp, *, repo_mock: MagicMock) -> MagicMock:
    """Monkeypatch ArtifactRepository so it returns *repo_mock* when instantiated."""
    mock_cls = MagicMock()
    mock_cls.return_value = repo_mock
    mp.setattr(
        "lovely_assistant.app.routes.artifacts.ArtifactRepository",
        mock_cls,
    )
    return mock_cls


# ---------------------------------------------------------------------------
# GET /artifacts
# ---------------------------------------------------------------------------


class TestListArtifacts:
    @pytest.mark.asyncio
    async def test_list_artifacts(self):
        rows = [
            _make_row(row_id=1, name="persona", version=1),
            _make_row(row_id=2, name="ecosystem", version=1),
        ]
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.get_all_active = AsyncMock(return_value=rows)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/artifacts")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "persona"
        assert data[1]["name"] == "ecosystem"
        assert data[0]["is_active"] is True

    @pytest.mark.asyncio
    async def test_list_artifacts_no_db(self):
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/artifacts")

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# GET /artifacts/{name}
# ---------------------------------------------------------------------------


class TestGetArtifact:
    @pytest.mark.asyncio
    async def test_get_artifact(self):
        row = _make_row(name="persona", content="You are Jarvis.")
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.get_active = AsyncMock(return_value=row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/artifacts/persona")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "persona"
        assert data["content"] == "You are Jarvis."
        assert data["version"] == 1
        assert data["is_active"] is True
        assert data["proposed_by"] == "system"
        assert data["created_at"] == _NOW.isoformat()

    @pytest.mark.asyncio
    async def test_get_artifact_not_found(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.get_active = AsyncMock(return_value=None)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/artifacts/nonexistent")

        assert response.status_code == 404
        assert "nonexistent" in response.json()["detail"]


# ---------------------------------------------------------------------------
# GET /artifacts/{name}/history
# ---------------------------------------------------------------------------


class TestGetArtifactHistory:
    @pytest.mark.asyncio
    async def test_get_artifact_history(self):
        rows = [
            _make_row(row_id=3, name="persona", version=3, is_active=True),
            _make_row(row_id=2, name="persona", version=2, is_active=False),
            _make_row(row_id=1, name="persona", version=1, is_active=False),
        ]
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.get_history = AsyncMock(return_value=rows)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.get("/artifacts/persona/history")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["version"] == 3
        assert data[0]["is_active"] is True
        assert data[2]["version"] == 1
        assert data[2]["is_active"] is False


# ---------------------------------------------------------------------------
# POST /artifacts/{name}/propose
# ---------------------------------------------------------------------------


class TestProposeArtifact:
    @pytest.mark.asyncio
    async def test_propose_artifact(self):
        proposed_row = _make_row(
            row_id=4,
            name="persona",
            version=2,
            is_active=False,
            content="Updated persona",
            proposed_by="dashboard",
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.propose = AsyncMock(return_value=proposed_row)
            repo.get_active = AsyncMock(return_value=_make_row(name="persona", version=1, is_active=True))
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/artifacts/persona/propose",
                    json={"content": "Updated persona"},
                )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "persona"
        assert data["version"] == 2
        assert data["is_active"] is False
        assert data["content"] == "Updated persona"
        assert data["proposed_by"] == "dashboard"
        assert data["success"] is True
        assert data["live_version"] == 1
        assert data["effective_on_next_request"] is False
        assert "approval" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_propose_artifact_validates_empty_content(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/propose",
                json={"content": ""},
            )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /artifacts/{name}/approve/{version}
# ---------------------------------------------------------------------------


class TestApproveArtifact:
    @pytest.mark.asyncio
    async def test_approve_artifact(self):
        approved_row = _make_row(
            row_id=4,
            name="persona",
            version=2,
            is_active=True,
            content="Updated persona",
            proposed_by="dashboard",
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.approve = AsyncMock(return_value=approved_row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post("/artifacts/persona/approve/2")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "persona"
        assert data["version"] == 2
        assert data["is_active"] is True
        assert data["success"] is True
        assert data["live_version"] == 2
        assert data["effective_on_next_request"] is True
        assert "approved" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_approve_artifact_not_found(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.approve = AsyncMock(return_value=None)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post("/artifacts/persona/approve/99")

        assert response.status_code == 404
        assert "99" in response.json()["detail"]
        assert "persona" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /artifacts/{name}/rollback/{version}
# ---------------------------------------------------------------------------


class TestRollbackArtifact:
    @pytest.mark.asyncio
    async def test_rollback_artifact(self):
        rolled_back_row = _make_row(
            row_id=1,
            name="persona",
            version=1,
            is_active=True,
            content="Original persona",
            proposed_by="system",
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.rollback = AsyncMock(return_value=rolled_back_row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post("/artifacts/persona/rollback/1")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "persona"
        assert data["version"] == 1
        assert data["is_active"] is True
        assert data["success"] is True
        assert data["live_version"] == 1
        assert data["effective_on_next_request"] is True
        assert "rolled back" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_rollback_artifact_not_found(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.rollback = AsyncMock(return_value=None)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post("/artifacts/persona/rollback/99")

        assert response.status_code == 404
        assert "99" in response.json()["detail"]
        assert "persona" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /artifacts/{name}/actions
# ---------------------------------------------------------------------------


class TestArtifactActions:
    @pytest.mark.asyncio
    async def test_approve_action(self):
        approved_row = _make_row(
            row_id=4, name="persona", version=2, is_active=True, content="Updated"
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.approve = AsyncMock(return_value=approved_row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/artifacts/persona/actions",
                    json={"action": "approve", "version": 2},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["version"] == 2
        assert data["is_active"] is True
        assert data["live_version"] == 2
        assert data["effective_on_next_request"] is True

    @pytest.mark.asyncio
    async def test_approve_missing_version(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/actions",
                json={"action": "approve"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_approve_not_found(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.approve = AsyncMock(return_value=None)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/artifacts/persona/actions",
                    json={"action": "approve", "version": 99},
                )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_rollback_action(self):
        rolled_back = _make_row(
            row_id=1, name="persona", version=1, is_active=True, content="Original"
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.rollback = AsyncMock(return_value=rolled_back)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/artifacts/persona/actions",
                    json={"action": "rollback", "version": 1},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["version"] == 1
        assert data["live_version"] == 1

    @pytest.mark.asyncio
    async def test_rollback_missing_version(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/actions",
                json={"action": "rollback"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_propose_action(self):
        proposed_row = _make_row(
            row_id=5, name="persona", version=3, is_active=False, content="New content"
        )
        active_row = _make_row(name="persona", version=2, is_active=True)
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.propose = AsyncMock(return_value=proposed_row)
            repo.get_active = AsyncMock(return_value=active_row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.post(
                    "/artifacts/persona/actions",
                    json={"action": "propose", "content": "New content"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["version"] == 3
        assert data["is_active"] is False
        assert data["live_version"] == 2
        assert data["effective_on_next_request"] is False

    @pytest.mark.asyncio
    async def test_propose_missing_content(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/actions",
                json={"action": "propose"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_action(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/actions",
                json={"action": "delete"},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_no_db(self):
        app = _make_app(db_service=None)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post(
                "/artifacts/persona/actions",
                json={"action": "approve", "version": 1},
            )

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# PATCH /artifacts/scratchpad
# ---------------------------------------------------------------------------


class TestUpdateScratchpad:
    @pytest.mark.asyncio
    async def test_update_scratchpad(self):
        updated_row = _make_row(
            row_id=5,
            name="scratchpad",
            version=3,
            is_active=True,
            content="Updated scratchpad notes",
            proposed_by="dashboard",
        )
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        with pytest.MonkeyPatch.context() as mp:
            repo = MagicMock()
            repo.update_scratchpad = AsyncMock(return_value=updated_row)
            _patch_repo(mp, repo_mock=repo)

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
                response = await c.patch(
                    "/artifacts/scratchpad",
                    json={"content": "Updated scratchpad notes"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "scratchpad"
        assert data["content"] == "Updated scratchpad notes"
        assert data["is_active"] is True
        assert data["success"] is True
        assert data["live_version"] == 3
        assert data["effective_on_next_request"] is True
        assert "activated immediately" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_update_scratchpad_validates_empty_content(self):
        mock_db = _make_mock_db()
        app = _make_app(db_service=mock_db)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.patch(
                "/artifacts/scratchpad",
                json={"content": ""},
            )

        assert response.status_code == 422
