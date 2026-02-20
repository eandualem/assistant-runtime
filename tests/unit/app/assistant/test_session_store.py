"""Tests for the session store — in-memory and DB persistence paths."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

from lovely_assistant.app.assistant._session_store import SessionStore


class TestSessionStore:
    def test_get_context_creates_new(self):
        store = SessionStore()
        ctx = store.get_context("s1")
        assert ctx["turn_number"] == 0
        assert ctx["working_memory"] is None
        assert ctx["pending_tool_call"] is None
        assert ctx["message_history"] == []

    def test_get_context_returns_existing(self):
        store = SessionStore()
        ctx1 = store.get_context("s1")
        ctx1["turn_number"] = 5
        ctx2 = store.get_context("s1")
        assert ctx2["turn_number"] == 5

    def test_has_session(self):
        store = SessionStore()
        assert store.has_session("s1") is False
        store.get_context("s1")
        assert store.has_session("s1") is True

    def test_session_count(self):
        store = SessionStore()
        assert store.session_count() == 0
        store.get_context("s1")
        store.get_context("s2")
        assert store.session_count() == 2

    def test_increment_turn(self):
        store = SessionStore()
        assert store.increment_turn("s1") == 1
        assert store.increment_turn("s1") == 2
        assert store.increment_turn("s1") == 3

    def test_get_history_empty(self):
        store = SessionStore()
        assert store.get_history("s1") == []

    def test_save_and_get_history(self):
        store = SessionStore()
        messages = [{"role": "user", "content": "hi"}]
        store.save_history("s1", messages)
        assert store.get_history("s1") == messages

    def test_set_pending_tool_call(self):
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-123", "ui_notify")
        ctx = store.get_context("s1")
        assert ctx["pending_tool_call"] == {
            "tool_call_id": "tc-123",
            "tool_name": "ui_notify",
        }

    def test_clear_pending_tool_call(self):
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-123", "ui_notify")
        pending = store.clear_pending_tool_call("s1")
        assert pending == {"tool_call_id": "tc-123", "tool_name": "ui_notify"}
        # Now cleared
        assert store.clear_pending_tool_call("s1") is None

    def test_clear_pending_when_none(self):
        store = SessionStore()
        store.get_context("s1")
        assert store.clear_pending_tool_call("s1") is None


# --- Helpers for DB tests ---


def _make_mock_db(*, healthy: bool = True) -> MagicMock:
    """Create a mock DatabaseService with configurable health."""
    mock_db = MagicMock()
    mock_db._healthy = healthy
    return mock_db


def _attach_session_context(mock_db: MagicMock, mock_async_session: AsyncMock) -> None:
    """Attach a working session_context async context manager to a mock DB."""

    @asynccontextmanager
    async def _session_context():
        yield mock_async_session

    mock_db.session_context = _session_context


# --- DB Persistence Tests ---


class TestSessionStoreWithDb:
    """Tests for async methods that interact with the DB persistence layer."""

    # --- get_context_async ---

    async def test_get_context_async_returns_from_memory_if_cached(self):
        """If session is already in memory, DB is not consulted."""
        mock_db = _make_mock_db()
        store = SessionStore(database_service=mock_db)

        # Pre-populate in-memory
        store.get_context("s1")["turn_number"] = 7

        ctx = await store.get_context_async("s1")
        assert ctx["turn_number"] == 7

    async def test_get_context_async_loads_from_db_on_miss(self):
        """On cache miss, _load_session_from_db is called and result is cached."""
        mock_db = _make_mock_db()
        store = SessionStore(database_service=mock_db)

        db_session_data = {
            "turn_number": 3,
            "working_memory": "some memory",
            "pending_tool_call": None,
            "message_history": [],
        }
        store._load_session_from_db = AsyncMock(return_value=db_session_data)

        ctx = await store.get_context_async("s1")

        store._load_session_from_db.assert_awaited_once_with("s1")
        assert ctx["turn_number"] == 3
        assert ctx["working_memory"] == "some memory"
        # Should now be cached in memory
        assert store.has_session("s1") is True

    async def test_get_context_async_creates_new_if_not_in_db(self):
        """If DB returns None, a fresh session context is created."""
        mock_db = _make_mock_db()
        store = SessionStore(database_service=mock_db)
        store._load_session_from_db = AsyncMock(return_value=None)

        ctx = await store.get_context_async("s1")

        store._load_session_from_db.assert_awaited_once_with("s1")
        assert ctx["turn_number"] == 0
        assert ctx["message_history"] == []

    async def test_get_context_async_falls_back_when_db_unhealthy(self):
        """When DB is unhealthy, skip DB load and create fresh context."""
        mock_db = _make_mock_db(healthy=False)
        store = SessionStore(database_service=mock_db)
        store._load_session_from_db = AsyncMock()

        ctx = await store.get_context_async("s1")

        # DB load should NOT be called when unhealthy
        store._load_session_from_db.assert_not_awaited()
        assert ctx["turn_number"] == 0

    # --- save_history_async ---

    async def test_save_history_async_calls_persist(self):
        """save_history_async persists to DB after saving in memory."""
        mock_db = _make_mock_db()
        store = SessionStore(database_service=mock_db)
        store._persist_to_db = AsyncMock()

        messages = [MagicMock()]
        await store.save_history_async("s1", messages)

        # In-memory history should be set
        assert store.get_history("s1") == messages
        # DB persist should be called
        store._persist_to_db.assert_awaited_once_with("s1")

    async def test_save_history_async_graceful_on_db_failure(self):
        """In-memory state is saved even when DB persistence fails internally.

        The real _persist_to_db catches exceptions internally (logger.warning),
        so save_history_async always completes. We test the real _persist_to_db
        path with a DB that raises on session_context.
        """
        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _failing_context():
            raise RuntimeError("DB write failed")
            yield  # noqa: F401

        mock_db.session_context = _failing_context

        store = SessionStore(database_service=mock_db)
        messages = [MagicMock()]

        # Should not raise — _persist_to_db swallows the exception
        await store.save_history_async("s1", messages)
        # In-memory state should still be saved
        assert store.get_history("s1") == messages

    # --- delete_session ---

    async def test_delete_session_removes_from_memory(self):
        """delete_session removes the session from in-memory dict."""
        store = SessionStore()
        store.get_context("s1")
        assert store.has_session("s1") is True

        await store.delete_session("s1")
        assert store.has_session("s1") is False

    async def test_delete_session_removes_from_db(self):
        """delete_session calls DB to delete when DB is healthy."""
        mock_db = _make_mock_db()
        mock_async_session = AsyncMock()
        _attach_session_context(mock_db, mock_async_session)

        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        # Mock the repository class used via inline import inside delete_session
        mock_repo = MagicMock()
        mock_repo.delete = AsyncMock()

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)

            await store.delete_session("s1")

            mock_repo.delete.assert_awaited_once_with("s1")
            assert store.has_session("s1") is False
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

    async def test_delete_session_graceful_on_db_failure(self):
        """delete_session still removes from memory when DB fails."""
        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _failing_context():
            raise RuntimeError("DB connection failed")
            yield  # noqa: F401

        mock_db.session_context = _failing_context

        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        # Should not raise
        await store.delete_session("s1")
        assert store.has_session("s1") is False

    # --- list_sessions ---

    async def test_list_sessions_returns_from_db(self):
        """list_sessions queries DB when healthy."""
        mock_db = _make_mock_db()
        mock_async_session = AsyncMock()
        _attach_session_context(mock_db, mock_async_session)

        store = SessionStore(database_service=mock_db)

        # Mock repository and rows
        mock_row = MagicMock()
        mock_row.id = "s1"
        mock_row.title = "Test Session"
        mock_row.turn_number = 3
        mock_row.message_history = [{"kind": "request"}]
        mock_row.created_at = MagicMock()
        mock_row.created_at.isoformat.return_value = "2026-02-20T10:00:00"

        mock_repo = MagicMock()
        mock_repo.list_all = AsyncMock(return_value=[mock_row])

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)

        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)

            result = await store.list_sessions(limit=10, offset=0)

            assert len(result) == 1
            assert result[0]["session_id"] == "s1"
            assert result[0]["title"] == "Test Session"
            assert result[0]["turn_number"] == 3
            assert result[0]["message_count"] == 1
            assert result[0]["created_at"] == "2026-02-20T10:00:00"
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

    async def test_list_sessions_falls_back_to_memory_when_no_db(self):
        """list_sessions returns in-memory sessions when DB is None."""
        store = SessionStore()
        store.get_context("s1")["turn_number"] = 2
        store.get_context("s2")["turn_number"] = 5

        result = await store.list_sessions()

        assert len(result) == 2
        session_ids = {s["session_id"] for s in result}
        assert session_ids == {"s1", "s2"}

    # --- _load_session_from_db ---

    async def test_load_session_from_db_returns_none_when_no_db(self):
        """_load_session_from_db returns None when DB is not configured."""
        store = SessionStore()
        result = await store._load_session_from_db("s1")
        assert result is None

    # --- _persist_to_db ---

    async def test_persist_to_db_noop_when_no_db(self):
        """_persist_to_db does nothing when DB is not configured."""
        store = SessionStore()
        store.get_context("s1")
        # Should not raise
        await store._persist_to_db("s1")

    async def test_persist_to_db_noop_when_db_unhealthy(self):
        """_persist_to_db does nothing when DB is unhealthy."""
        mock_db = _make_mock_db(healthy=False)
        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        # Should not raise, and should not attempt DB access
        await store._persist_to_db("s1")
        # If it tried to access DB, session_context would fail since it's not set up
