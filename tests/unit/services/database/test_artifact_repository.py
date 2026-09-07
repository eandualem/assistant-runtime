"""Tests for ArtifactRepository — versioned artifact CRUD with mocked AsyncSession."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from assistant_runtime.services.database.models import ArtifactORM
from assistant_runtime.services.database.repositories import ArtifactRepository

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

    @asynccontextmanager
    async def savepoint():
        yield

    session.begin_nested = MagicMock(side_effect=savepoint)
    return session


@pytest.fixture
def repo(mock_session):
    return ArtifactRepository(session=mock_session)


# ---------------------------------------------------------------------------
# get_active
# ---------------------------------------------------------------------------


class TestGetActive:
    async def test_get_active_returns_active_version(self, repo, mock_session):
        mock_row = MagicMock(spec=ArtifactORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await repo.get_active("default", "persona")

        assert result is mock_row
        mock_result.scalar_one_or_none.assert_called_once()

    async def test_get_active_returns_none_when_missing(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get_active("default", "nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# get_all_active
# ---------------------------------------------------------------------------


class TestGetAllActive:
    async def test_get_all_active_returns_all(self, repo, mock_session):
        mock_rows = [MagicMock(spec=ArtifactORM), MagicMock(spec=ArtifactORM)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.get_all_active("default")

        assert result == mock_rows
        assert isinstance(result, list)
        assert len(result) == 2

    async def test_get_all_active_returns_empty_list(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.get_all_active("default")

        assert result == []


# ---------------------------------------------------------------------------
# get_history
# ---------------------------------------------------------------------------


class TestGetHistory:
    async def test_get_history_ordered_by_version_desc(self, repo, mock_session):
        mock_rows = [MagicMock(spec=ArtifactORM), MagicMock(spec=ArtifactORM)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.get_history("default", "persona")

        assert result == mock_rows
        mock_session.execute.assert_awaited_once()

    async def test_get_history_returns_empty_for_unknown_name(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.get_history("default", "nonexistent")

        assert result == []


# ---------------------------------------------------------------------------
# propose
# ---------------------------------------------------------------------------


class TestPropose:
    async def test_propose_increments_version(self, repo, mock_session):
        # First call: MAX(version) query returns 2
        max_result = MagicMock()
        max_result.scalar_one_or_none.return_value = 2
        created_row = ArtifactORM(
            name="persona",
            content="new content",
            version=3,
            is_active=False,
            proposed_by="operator",
        )
        insert_result = MagicMock()
        insert_result.scalar_one.return_value = created_row
        mock_session.execute.side_effect = [max_result, insert_result]

        result = await repo.propose("default", "persona", "new content", "operator")

        assert isinstance(result, ArtifactORM)
        assert result.version == 3
        assert result.name == "persona"
        assert result.content == "new content"
        assert result.proposed_by == "operator"
        assert result.is_active is False
        assert mock_session.execute.await_count == 2
        mock_session.flush.assert_awaited_once()

    async def test_propose_first_version_is_1(self, repo, mock_session):
        # MAX(version) returns None when no rows exist for this name
        max_result = MagicMock()
        max_result.scalar_one_or_none.return_value = None
        created_row = ArtifactORM(
            name="brand_new",
            content="initial content",
            version=1,
            is_active=False,
            proposed_by="assistant",
        )
        insert_result = MagicMock()
        insert_result.scalar_one.return_value = created_row
        mock_session.execute.side_effect = [max_result, insert_result]

        result = await repo.propose("default", "brand_new", "initial content", "assistant")

        assert result.version == 1
        assert result.name == "brand_new"
        assert result.is_active is False

    async def test_propose_does_not_activate(self, repo, mock_session):
        max_result = MagicMock()
        max_result.scalar_one_or_none.return_value = 5
        created_row = ArtifactORM(
            name="persona",
            content="draft",
            version=6,
            is_active=False,
            proposed_by="assistant",
        )
        insert_result = MagicMock()
        insert_result.scalar_one.return_value = created_row
        mock_session.execute.side_effect = [max_result, insert_result]

        result = await repo.propose("default", "persona", "draft", "assistant")

        assert result.is_active is False


# ---------------------------------------------------------------------------
# approve
# ---------------------------------------------------------------------------


class TestApprove:
    async def test_approve_deactivates_previous(self, repo, mock_session):
        target_row = ArtifactORM(
            id=11,
            name="persona",
            content="v3",
            version=3,
            is_active=True,
            proposed_by="operator",
        )

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = 11
        update_result = MagicMock()
        activate_result = MagicMock()
        activate_result.scalar_one.return_value = target_row

        mock_session.execute.side_effect = [select_result, update_result, activate_result]

        result = await repo.approve("default", "persona", version=3)

        assert result is target_row
        assert mock_session.execute.await_count == 3
        mock_session.flush.assert_awaited_once()

    async def test_approve_returns_none_for_missing_version(self, repo, mock_session):
        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = select_result

        result = await repo.approve("default", "persona", version=999)

        assert result is None
        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_not_awaited()

    async def test_approve_activates_target(self, repo, mock_session):
        target_row = ArtifactORM(
            id=9,
            name="persona",
            content="v2",
            version=2,
            is_active=True,
            proposed_by="operator",
        )

        select_result = MagicMock()
        select_result.scalar_one_or_none.return_value = 9
        update_result = MagicMock()
        activate_result = MagicMock()
        activate_result.scalar_one.return_value = target_row
        mock_session.execute.side_effect = [select_result, update_result, activate_result]

        result = await repo.approve("default", "persona", version=2)

        assert result.is_active is True


class TestProposeRetry:
    async def test_concurrent_version_collision_is_retried(self, repo, mock_session):
        max_result = MagicMock()
        max_result.scalar_one_or_none.return_value = 2
        inserted = MagicMock(spec=ArtifactORM)
        inserted.version = 4
        insert_result = MagicMock()
        insert_result.scalar_one.return_value = inserted
        max_after = MagicMock()
        max_after.scalar_one_or_none.return_value = 3
        mock_session.execute.side_effect = [
            max_result,
            IntegrityError("insert", {}, Exception("duplicate")),
            max_after,
            insert_result,
        ]

        result = await repo.propose("default", "persona", "text", "host")

        assert result is inserted
        assert mock_session.begin_nested.call_count == 2
        second_insert = mock_session.execute.call_args_list[3].args[0]
        assert second_insert.compile().params["version"] == 4

    async def test_persistent_collision_raises(self, repo, mock_session):
        max_result = MagicMock()
        max_result.scalar_one_or_none.return_value = 1
        mock_session.execute.side_effect = [
            max_result,
            IntegrityError("insert", {}, Exception("dup")),
        ] * 3

        with pytest.raises(IntegrityError):
            await repo.propose("default", "persona", "text", "host")
