"""Tests for the artifact management tool."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lovely_assistant.services.tools._artifact_tools import manage_artifacts

MODULE = "lovely_assistant.services.database.repositories"


@pytest.fixture(autouse=True)
def _cleanup_deps():
    """Remove _handler_deps after each test to avoid cross-test pollution."""
    yield
    if hasattr(manage_artifacts, "_handler_deps"):
        del manage_artifacts._handler_deps


def _make_mock_db():
    """Create a mock database service with a session_context async context manager."""
    mock_db = MagicMock()

    @asynccontextmanager
    async def mock_session_ctx():
        yield MagicMock()

    mock_db.session_context = mock_session_ctx
    return mock_db


def _make_artifact_row(
    *,
    name: str = "persona",
    content: str = "You are Jarvis.",
    version: int = 1,
    is_active: bool = True,
    proposed_by: str = "system",
    created_at: datetime | None = None,
) -> MagicMock:
    """Create a mock ArtifactORM row with the given attributes."""
    row = MagicMock()
    row.name = name
    row.content = content
    row.version = version
    row.is_active = is_active
    row.proposed_by = proposed_by
    row.created_at = created_at or datetime(2026, 2, 22, 12, 0, 0, tzinfo=UTC)
    return row


class TestManageArtifactsDispatch:
    """Tests for action dispatch and dependency validation."""

    async def test_unknown_action_returns_error(self):
        result = await manage_artifacts(action="invalid")
        assert result["success"] is False
        assert "Unknown action" in result["error"]
        assert "invalid" in result["error"]

    async def test_no_deps_returns_error(self):
        result = await manage_artifacts(action="list")
        assert result["success"] is False
        assert "not available" in result["error"]

    async def test_no_database_service_returns_error(self):
        manage_artifacts._handler_deps = {"database_service": None}
        result = await manage_artifacts(action="list")
        assert result["success"] is False
        assert "not available" in result["error"]


class TestListAction:
    """Tests for the list action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_list_action_returns_artifacts(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        rows = [
            _make_artifact_row(name="ecosystem", content="Agent ecosystem info", version=3),
            _make_artifact_row(name="persona", content="You are Jarvis.", version=2),
            _make_artifact_row(name="scratchpad", content="Notes here", version=5),
        ]
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_all_active = AsyncMock(return_value=rows)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="list")

        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["artifacts"]) == 3

        artifact_names = [a["name"] for a in result["artifacts"]]
        assert "ecosystem" in artifact_names
        assert "persona" in artifact_names
        assert "scratchpad" in artifact_names

        # Check individual artifact structure
        persona = next(a for a in result["artifacts"] if a["name"] == "persona")
        assert persona["version"] == 2
        assert persona["char_count"] == len("You are Jarvis.")
        assert persona["proposed_by"] == "system"
        assert persona["created_at"] is not None


class TestViewAction:
    """Tests for the view action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_view_action_returns_content(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        row = _make_artifact_row(name="persona", content="You are Jarvis.", version=2)
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_active = AsyncMock(return_value=row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="view", name="persona")

        assert result["success"] is True
        assert result["name"] == "persona"
        assert result["content"] == "You are Jarvis."
        assert result["version"] == 2
        assert result["proposed_by"] == "system"
        assert result["created_at"] is not None

    async def test_view_action_no_name_returns_error(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="view", name="")

        assert result["success"] is False
        assert "Name is required" in result["error"]

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_view_action_not_found_returns_error(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        mock_repo_instance = MagicMock()
        mock_repo_instance.get_active = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="view", name="nonexistent")

        assert result["success"] is False
        assert "No active artifact found" in result["error"]
        assert "nonexistent" in result["error"]


class TestProposeEditAction:
    """Tests for the propose_edit action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_propose_edit_creates_inactive_version(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        new_row = _make_artifact_row(
            name="persona",
            content="Updated persona.",
            version=3,
            is_active=False,
            proposed_by="jarvis",
        )
        mock_repo_instance = MagicMock()
        mock_repo_instance.propose = AsyncMock(return_value=new_row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(
            action="propose_edit", name="persona", content="Updated persona."
        )

        assert result["success"] is True
        assert result["name"] == "persona"
        assert result["version"] == 3
        assert result["is_active"] is False
        assert "approval" in result["message"].lower()

        # Verify propose was called with correct args
        mock_repo_instance.propose.assert_awaited_once_with(
            "persona", "Updated persona.", proposed_by="jarvis"
        )

    async def test_propose_edit_no_name_returns_error(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="propose_edit", name="", content="some content")

        assert result["success"] is False
        assert "Name is required" in result["error"]

    async def test_propose_edit_no_content_returns_error(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="propose_edit", name="persona", content="")

        assert result["success"] is False
        assert "Content is required" in result["error"]


class TestUpdateScratchpadAction:
    """Tests for the update_scratchpad action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_update_scratchpad_auto_approves(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        row = _make_artifact_row(
            name="scratchpad",
            content="New observations.",
            version=4,
            is_active=True,
            proposed_by="jarvis",
        )
        mock_repo_instance = MagicMock()
        mock_repo_instance.update_scratchpad = AsyncMock(return_value=row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="update_scratchpad", content="New observations.")

        assert result["success"] is True
        assert result["name"] == "scratchpad"
        assert result["version"] == 4
        assert result["is_active"] is True
        assert "activated" in result["message"].lower()

        # Verify update_scratchpad was called with correct args
        mock_repo_instance.update_scratchpad.assert_awaited_once_with(
            "New observations.", proposed_by="jarvis"
        )

    async def test_update_scratchpad_no_content_returns_error(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="update_scratchpad", content="")

        assert result["success"] is False
        assert "Content is required" in result["error"]


class TestApproveAction:
    """Tests for the approve action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_approve_success(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        row = _make_artifact_row(
            name="persona",
            content="Approved persona.",
            version=3,
            is_active=True,
            proposed_by="jarvis",
        )
        mock_repo_instance = MagicMock()
        mock_repo_instance.approve = AsyncMock(return_value=row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="approve", name="persona", version=3)

        assert result["success"] is True
        assert result["name"] == "persona"
        assert result["version"] == 3
        assert result["is_active"] is True
        assert "active" in result["message"].lower()

        mock_repo_instance.approve.assert_awaited_once_with("persona", 3)

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_approve_version_not_found(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        mock_repo_instance = MagicMock()
        mock_repo_instance.approve = AsyncMock(return_value=None)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="approve", name="persona", version=99)

        assert result["success"] is False
        assert "No version 99" in result["error"]
        assert "persona" in result["error"]

    async def test_approve_missing_name(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="approve", name="", version=2)

        assert result["success"] is False
        assert "Name is required" in result["error"]

    async def test_approve_missing_version(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="approve", name="persona", version=0)

        assert result["success"] is False
        assert "Version is required" in result["error"]


class TestHistoryAction:
    """Tests for the history action."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_history_returns_versions(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        rows = [
            _make_artifact_row(name="persona", content="V1 content", version=1, is_active=False),
            _make_artifact_row(name="persona", content="V2 updated", version=2, is_active=True),
            _make_artifact_row(name="persona", content="V3 proposed", version=3, is_active=False),
        ]
        mock_repo_instance = MagicMock()
        mock_repo_instance.get_history = AsyncMock(return_value=rows)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="history", name="persona")

        assert result["success"] is True
        assert result["name"] == "persona"
        assert result["count"] == 3
        assert len(result["versions"]) == 3

        # Check structure of version entries
        v2 = result["versions"][1]
        assert v2["version"] == 2
        assert v2["is_active"] is True
        assert v2["proposed_by"] == "system"
        assert v2["char_count"] == len("V2 updated")
        assert v2["created_at"] is not None

        # Verify inactive version
        v1 = result["versions"][0]
        assert v1["is_active"] is False

    async def test_history_missing_name(self):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        result = await manage_artifacts(action="history", name="")

        assert result["success"] is False
        assert "Name is required" in result["error"]

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_history_empty(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        mock_repo_instance = MagicMock()
        mock_repo_instance.get_history = AsyncMock(return_value=[])
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="history", name="nonexistent")

        assert result["success"] is True
        assert result["count"] == 0
        assert result["versions"] == []


class TestDispatchApproveAndHistory:
    """Integration tests verifying action dispatch to approve and history handlers."""

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_action_approve_dispatches_correctly(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        row = _make_artifact_row(name="ecosystem", version=5, is_active=True)
        mock_repo_instance = MagicMock()
        mock_repo_instance.approve = AsyncMock(return_value=row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="approve", name="ecosystem", version=5)

        assert result["success"] is True
        assert result["version"] == 5
        mock_repo_instance.approve.assert_awaited_once_with("ecosystem", 5)

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_action_history_dispatches_correctly(self, mock_repo_cls):
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        mock_repo_instance = MagicMock()
        mock_repo_instance.get_history = AsyncMock(return_value=[])
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="history", name="persona")

        assert result["success"] is True
        mock_repo_instance.get_history.assert_awaited_once()

    @patch(f"{MODULE}.ArtifactRepository")
    async def test_version_param_passes_through_to_approve(self, mock_repo_cls):
        """Verify the version parameter from manage_artifacts reaches _approve_artifact."""
        mock_db = _make_mock_db()
        manage_artifacts._handler_deps = {"database_service": mock_db}

        row = _make_artifact_row(name="persona", version=7, is_active=True)
        mock_repo_instance = MagicMock()
        mock_repo_instance.approve = AsyncMock(return_value=row)
        mock_repo_cls.return_value = mock_repo_instance

        result = await manage_artifacts(action="approve", name="persona", version=7)

        assert result["success"] is True
        mock_repo_instance.approve.assert_awaited_once_with("persona", 7)
