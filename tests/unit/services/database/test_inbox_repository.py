"""Tests for InboxRepository — CRUD operations with mocked AsyncSession."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.services.database.models import InboxItemORM
from assistant_runtime.services.database.repositories import InboxRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_session():
    """AsyncSession mock with common async methods."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def repo(mock_session):
    return InboxRepository(session=mock_session)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    async def test_creates_inbox_item(self, repo, mock_session):
        created_row = InboxItemORM(from_agent="leo", message="Hello", severity="info", context=None)
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        result = await repo.create(from_agent="leo", message="Hello")

        assert isinstance(result, InboxItemORM)
        assert result.from_agent == "leo"
        assert result.message == "Hello"

    async def test_executes_insert_and_flushes(self, repo, mock_session):
        created_row = InboxItemORM(from_agent="leo", message="Hello", severity="info", context=None)
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        await repo.create(from_agent="leo", message="Hello")

        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()

    async def test_default_severity_is_info(self, repo):
        mock_session = repo._session
        created_row = InboxItemORM(from_agent="leo", message="Hello", severity="info", context=None)
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        result = await repo.create(from_agent="leo", message="Hello")

        assert result.severity == "info"

    async def test_accepts_severity_parameter(self, repo):
        mock_session = repo._session
        created_row = InboxItemORM(
            from_agent="leo", message="Alert!", severity="urgent", context=None
        )
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        result = await repo.create(from_agent="leo", message="Alert!", severity="urgent")

        assert result.severity == "urgent"

    async def test_accepts_context_parameter(self, repo):
        ctx = {"issue_number": 42, "repo": "orchestration"}
        mock_session = repo._session
        created_row = InboxItemORM(
            from_agent="leo", message="Check this", severity="info", context=ctx
        )
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        result = await repo.create(from_agent="leo", message="Check this", context=ctx)

        assert result.context == ctx

    async def test_context_defaults_to_none(self, repo):
        mock_session = repo._session
        created_row = InboxItemORM(from_agent="leo", message="Hello", severity="info", context=None)
        mock_result = MagicMock()
        mock_result.scalar_one.return_value = created_row
        mock_session.execute.return_value = mock_result

        result = await repo.create(from_agent="leo", message="Hello")

        assert result.context is None


# ---------------------------------------------------------------------------
# List Unsurfaced
# ---------------------------------------------------------------------------


class TestListUnsurfaced:
    async def test_returns_list_from_scalars(self, repo, mock_session):
        mock_rows = [MagicMock(spec=InboxItemORM), MagicMock(spec=InboxItemORM)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.list_unsurfaced()

        assert result == mock_rows
        assert isinstance(result, list)

    async def test_returns_empty_list_when_no_items(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.list_unsurfaced()

        assert result == []

    async def test_respects_limit(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        await repo.list_unsurfaced(limit=5)

        mock_session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Mark Surfaced
# ---------------------------------------------------------------------------


class TestMarkSurfaced:
    async def test_marks_item_as_surfaced(self, repo, mock_session):
        mock_row = InboxItemORM(from_agent="leo", message="Hello", severity="info", context=None)
        mock_row.surfaced = True
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await repo.mark_surfaced("item-123")

        assert result is mock_row
        assert mock_row.surfaced is True
        mock_session.flush.assert_awaited_once()

    async def test_returns_none_when_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.mark_surfaced("nonexistent")

        assert result is None
        mock_session.flush.assert_not_awaited()


# ---------------------------------------------------------------------------
# List All
# ---------------------------------------------------------------------------


class TestListAll:
    async def test_returns_list_from_scalars(self, repo, mock_session):
        mock_rows = [MagicMock(spec=InboxItemORM), MagicMock(spec=InboxItemORM)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.list_all()

        assert result == mock_rows
        assert isinstance(result, list)

    async def test_respects_limit_and_offset(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        await repo.list_all(limit=10, offset=20)

        mock_session.execute.assert_awaited_once()
