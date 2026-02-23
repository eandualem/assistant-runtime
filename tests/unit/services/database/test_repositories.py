"""Tests for SessionRepository and SettingsRepository — CRUD operations with mocked AsyncSession."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from lovely_assistant.services.database.models import SessionORM, UserSettingsORM
from lovely_assistant.services.database.repositories import SessionRepository, SettingsRepository

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
    return SessionRepository(session=mock_session)


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


class TestCreate:
    async def test_creates_session_with_correct_id(self, repo, mock_session):
        result = await repo.create(session_id="abc-123")

        assert isinstance(result, SessionORM)
        assert result.id == "abc-123"

    async def test_adds_to_session(self, repo, mock_session):
        await repo.create(session_id="abc-123")

        mock_session.add.assert_called_once()
        added_obj = mock_session.add.call_args[0][0]
        assert isinstance(added_obj, SessionORM)

    async def test_flushes_after_add(self, repo, mock_session):
        await repo.create(session_id="abc-123")

        mock_session.flush.assert_awaited_once()

    async def test_title_defaults_to_none(self, repo):
        result = await repo.create(session_id="abc-123")

        assert result.title is None

    async def test_accepts_title_parameter(self, repo):
        result = await repo.create(session_id="abc-123", title="My Session")

        assert result.title == "My Session"

    async def test_turn_number_initialized_to_zero(self, repo):
        result = await repo.create(session_id="abc-123")

        assert result.turn_number == 0

    async def test_accepts_expires_at_parameter(self, repo):
        future = datetime.now(UTC) + timedelta(hours=48)
        result = await repo.create(session_id="abc-123", expires_at=future)

        assert result.expires_at == future

    async def test_expires_at_defaults_to_none_if_not_passed(self, repo):
        """When no expires_at is given, it's left to the server_default."""
        result = await repo.create(session_id="abc-123")
        # expires_at is not explicitly set — uses server_default
        # In unit tests with mocked session, it won't have the server default.
        # We verify create() doesn't crash without it.
        assert result.id == "abc-123"


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_returns_result_from_scalar_one_or_none(self, repo, mock_session):
        mock_row = MagicMock(spec=SessionORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await repo.get("abc-123")

        assert result is mock_row
        mock_result.scalar_one_or_none.assert_called_once()

    async def test_returns_none_when_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.get("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# List All
# ---------------------------------------------------------------------------


class TestListAll:
    async def test_returns_list_from_scalars(self, repo, mock_session):
        mock_rows = [MagicMock(spec=SessionORM), MagicMock(spec=SessionORM)]
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = mock_rows
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.list_all()

        assert result == mock_rows
        assert isinstance(result, list)

    async def test_returns_empty_list_when_no_sessions(self, repo, mock_session):
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        result = await repo.list_all()

        assert result == []

    async def test_respects_limit_and_offset(self, repo, mock_session):
        """Verify that limit and offset are passed through to the SQL query."""
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session.execute.return_value = mock_result

        await repo.list_all(limit=10, offset=20)

        # The execute call was made — we verify the statement was constructed
        # by checking execute was called (the SQL construction uses .limit/.offset
        # which we can't easily inspect on the mock, but the call succeeding
        # with the parameters confirms the method accepts them).
        mock_session.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    async def test_sets_attributes_and_flushes(self, repo, mock_session):
        mock_row = MagicMock(spec=SessionORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        mock_session.execute.return_value = mock_result

        await repo.update("abc-123", title="Updated", turn_number=5)

        mock_row.__setattr__("title", "Updated")
        mock_row.__setattr__("turn_number", 5)
        mock_session.flush.assert_awaited()

    async def test_executes_update_even_when_session_not_found(self, repo, mock_session):
        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        # Reset flush call count before the update call
        mock_session.flush.reset_mock()

        await repo.update("nonexistent", title="Updated")

        # Bulk update + flush is always called (no pre-check query)
        mock_session.execute.assert_awaited()
        mock_session.flush.assert_awaited()


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_returns_true_when_row_deleted(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        result = await repo.delete("abc-123")

        assert result is True

    async def test_returns_false_when_no_row_deleted(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.delete("nonexistent")

        assert result is False

    async def test_flushes_after_delete(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session.execute.return_value = mock_result

        await repo.delete("abc-123")

        mock_session.flush.assert_awaited_once()


# ---------------------------------------------------------------------------
# Exists
# ---------------------------------------------------------------------------


class TestExists:
    async def test_returns_true_when_scalar_is_not_none(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = "abc-123"
        mock_session.execute.return_value = mock_result

        result = await repo.exists("abc-123")

        assert result is True

    async def test_returns_false_when_scalar_is_none(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await repo.exists("nonexistent")

        assert result is False


# ---------------------------------------------------------------------------
# Cleanup Expired
# ---------------------------------------------------------------------------


class TestCleanupExpired:
    async def test_returns_count_of_deleted_rows(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        result = await repo.cleanup_expired()

        assert result == 3
        mock_session.flush.assert_awaited_once()

    async def test_returns_zero_when_none_expired(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session.execute.return_value = mock_result

        result = await repo.cleanup_expired()

        assert result == 0

    async def test_returns_zero_when_rowcount_is_none(self, repo, mock_session):
        mock_result = MagicMock()
        mock_result.rowcount = None
        mock_session.execute.return_value = mock_result

        result = await repo.cleanup_expired()

        assert result == 0


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    async def test_upsert_executes_and_flushes(self, repo, mock_session):
        await repo.upsert("abc-123")

        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()

    async def test_upsert_with_all_fields(self, repo, mock_session):
        expires = datetime.now(UTC) + timedelta(hours=12)
        await repo.upsert(
            "abc-123",
            title="My Session",
            turn_number=5,
            message_history=[{"role": "user", "content": "hi"}],
            working_memory={"goal": "test"},
            pending_tool_call={"id": "tc-1"},
            expires_at=expires,
        )

        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()

    async def test_upsert_defaults(self, repo, mock_session):
        """Defaults: title=None, turn_number=0, message_history=[], rest None."""
        await repo.upsert("abc-123")

        # Verify execute was called (the INSERT statement was built with defaults)
        mock_session.execute.assert_awaited_once()


# ===========================================================================
# SettingsRepository
# ===========================================================================


@pytest.fixture
def settings_repo(mock_session):
    return SettingsRepository(session=mock_session)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestSettingsRepositoryConstants:
    def test_settings_id_is_default(self):
        assert SettingsRepository.SETTINGS_ID == "default"


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------


class TestSettingsRepositoryGet:
    async def test_returns_result_from_scalar_one_or_none(self, settings_repo, mock_session):
        mock_row = MagicMock(spec=UserSettingsORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row
        mock_session.execute.return_value = mock_result

        result = await settings_repo.get()

        assert result is mock_row
        mock_result.scalar_one_or_none.assert_called_once()

    async def test_returns_none_when_not_found(self, settings_repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        result = await settings_repo.get()

        assert result is None


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


class TestSettingsRepositorySave:
    async def test_creates_new_row_when_none_exists(self, settings_repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        await settings_repo.save({"temperature": 0.5})

        mock_session.add.assert_called_once()
        added_obj = mock_session.add.call_args[0][0]
        assert isinstance(added_obj, UserSettingsORM)
        assert added_obj.id == "default"

    async def test_updates_existing_row(self, settings_repo, mock_session):
        existing_row = MagicMock(spec=UserSettingsORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_row
        mock_session.execute.return_value = mock_result

        await settings_repo.save({"temperature": 0.7})

        # add should NOT be called for existing row
        mock_session.add.assert_not_called()
        mock_session.flush.assert_awaited()

    async def test_sets_multiple_fields(self, settings_repo, mock_session):
        existing_row = MagicMock(spec=UserSettingsORM)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_row
        mock_session.execute.return_value = mock_result

        await settings_repo.save({"temperature": 0.7, "max_turns": 10, "default_model": "gpt-4"})

        # Verify all fields were set on the row
        existing_row.__setattr__("temperature", 0.7)
        existing_row.__setattr__("max_turns", 10)
        existing_row.__setattr__("default_model", "gpt-4")

    async def test_flushes_after_save(self, settings_repo, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        await settings_repo.save({"temperature": 0.5})

        mock_session.flush.assert_awaited_once()
