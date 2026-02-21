"""Tests for the session store — in-memory and DB persistence paths."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from lovely_assistant.app.assistant._session_store import (
    _MAX_MEMORY_SESSIONS,
    SessionStore,
)


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
        store.set_pending_tool_call("s1", "tc-123", "navigate")
        ctx = store.get_context("s1")
        pending = ctx["pending_tool_call"]
        assert pending["tool_call_id"] == "tc-123"
        assert pending["tool_name"] == "navigate"
        assert "created_at" in pending

    def test_clear_pending_tool_call(self):
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-123", "navigate")
        pending = store.clear_pending_tool_call("s1")
        assert pending["tool_call_id"] == "tc-123"
        assert pending["tool_name"] == "navigate"
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

    async def test_get_context_async_attempts_db_even_when_unhealthy(self):
        """DB load is attempted regardless of _healthy flag — exception handling is the gate."""
        mock_db = _make_mock_db(healthy=False)
        store = SessionStore(database_service=mock_db)
        store._load_session_from_db = AsyncMock(return_value=None)

        ctx = await store.get_context_async("s1")

        # DB load IS called — _healthy is not a gate
        store._load_session_from_db.assert_awaited_once_with("s1")
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

    # --- save_history_async auto-title ---

    async def test_save_history_async_auto_titles_from_first_user_message(self):
        """Session title is auto-set from the first user message."""
        store = SessionStore()
        messages = [
            ModelRequest(parts=[UserPromptPart(content="What agents are idle right now?")]),
        ]
        await store.save_history_async("s1", messages)

        ctx = store.get_context("s1")
        assert ctx["title"] == "What agents are idle right now?"

    async def test_save_history_async_truncates_long_title(self):
        """Long first messages are truncated to 50 chars with ellipsis."""
        store = SessionStore()
        long_text = "A" * 80
        messages = [
            ModelRequest(parts=[UserPromptPart(content=long_text)]),
        ]
        await store.save_history_async("s1", messages)

        ctx = store.get_context("s1")
        assert ctx["title"] == "A" * 50 + "..."
        assert len(ctx["title"]) == 53

    async def test_save_history_async_does_not_overwrite_existing_title(self):
        """Once a title is set, subsequent saves don't overwrite it."""
        store = SessionStore()
        ctx = store.get_context("s1")
        ctx["title"] = "My custom title"

        messages = [
            ModelRequest(parts=[UserPromptPart(content="New first message")]),
        ]
        await store.save_history_async("s1", messages)

        assert ctx["title"] == "My custom title"

    async def test_save_history_async_skips_title_for_empty_messages(self):
        """No title is set when there are no messages."""
        store = SessionStore()
        await store.save_history_async("s1", [])

        ctx = store.get_context("s1")
        assert ctx.get("title") is None

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

    async def test_delete_session_propagates_db_error(self):
        """delete_session raises when DB fails — session stays in memory."""
        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _failing_context():
            raise RuntimeError("DB connection failed")
            yield  # noqa: F401

        mock_db.session_context = _failing_context

        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        with pytest.raises(RuntimeError, match="DB connection failed"):
            await store.delete_session("s1")
        # Memory pop only happens after DB succeeds — session is still in memory
        assert store.has_session("s1") is True

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

    async def test_list_sessions_propagates_db_error(self):
        """list_sessions raises when DB is configured but broken."""
        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _failing_context():
            raise RuntimeError("DB query failed")
            yield  # noqa: F401

        mock_db.session_context = _failing_context

        store = SessionStore(database_service=mock_db)

        with pytest.raises(RuntimeError, match="DB query failed"):
            await store.list_sessions()

    # --- _load_session_from_db ---

    async def test_load_session_from_db_returns_none_when_no_db(self):
        """_load_session_from_db returns None when DB is not configured."""
        store = SessionStore()
        result = await store._load_session_from_db("s1")
        assert result is None

    async def test_load_session_from_db_deserialization_failure_returns_empty_history(self):
        """When message history fails to deserialize, session loads with empty history."""
        mock_db = _make_mock_db()
        store = SessionStore(database_service=mock_db)

        mock_row = MagicMock()
        mock_row.turn_number = 3
        mock_row.working_memory = None
        mock_row.pending_tool_call = None
        mock_row.message_history = [{"bad": "data", "not_a_real": "message"}]
        mock_row.title = "Test session"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_row

        mock_async_session = AsyncMock()
        mock_async_session.execute = AsyncMock(return_value=mock_result)

        @asynccontextmanager
        async def _ctx():
            yield mock_async_session

        mock_db.session_context = _ctx

        result = await store._load_session_from_db("s1")
        assert result is not None
        assert result["turn_number"] == 3
        assert result["title"] == "Test session"
        assert result["message_history"] == []  # Empty — deserialization failed gracefully

    # --- _persist_to_db ---

    async def test_persist_to_db_noop_when_no_db(self):
        """_persist_to_db does nothing when DB is not configured."""
        store = SessionStore()
        store.get_context("s1")
        # Should not raise
        await store._persist_to_db("s1")

    async def test_persist_to_db_graceful_when_db_unhealthy(self):
        """_persist_to_db attempts DB access but handles failure gracefully."""
        mock_db = _make_mock_db(healthy=False)
        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        # Should not raise — exception is caught internally
        await store._persist_to_db("s1")


class TestPersistToDbFullPath:
    """Tests for _persist_to_db end-to-end: upsert-based persistence."""

    async def test_persist_calls_upsert(self):
        """_persist_to_db calls repo.upsert() with all session fields."""
        from pydantic_ai.messages import ModelResponse, TextPart

        mock_db = MagicMock()
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _ctx():
            yield mock_async_session

        mock_db.session_context = _ctx

        store = SessionStore(database_service=mock_db)
        # Pre-populate session
        ctx = store.get_context("s1")
        ctx["turn_number"] = 2
        ctx["title"] = "Test session"
        ctx["message_history"] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi!")]),
        ]

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)
            await store._persist_to_db("s1")
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        mock_repo.upsert.assert_awaited_once()
        call_kwargs = mock_repo.upsert.call_args
        assert call_kwargs[0][0] == "s1"
        assert call_kwargs[1]["title"] == "Test session"
        assert call_kwargs[1]["turn_number"] == 2
        assert "expires_at" in call_kwargs[1]
        assert isinstance(call_kwargs[1]["message_history"], list)
        assert len(call_kwargs[1]["message_history"]) == 2
        # get() and create() should NOT be called
        assert not hasattr(mock_repo, "get") or not mock_repo.get.called
        assert not hasattr(mock_repo, "create") or not mock_repo.create.called

    async def test_persist_converts_pydantic_model_to_dict(self):
        """WorkingMemory Pydantic model is converted to dict before upsert."""
        from lovely_assistant.services.history.models import WorkingMemory

        mock_db = MagicMock()
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _ctx():
            yield mock_async_session

        mock_db.session_context = _ctx

        store = SessionStore(database_service=mock_db)
        ctx = store.get_context("s1")
        ctx["working_memory"] = WorkingMemory(active_goal="Test goal", progress="In progress")

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)
            await store._persist_to_db("s1")
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        # working_memory should be a dict, not a Pydantic model
        call_kwargs = mock_repo.upsert.call_args[1]
        assert isinstance(call_kwargs["working_memory"], dict)
        assert call_kwargs["working_memory"]["active_goal"] == "Test goal"


# --- TTL and Eviction Tests ---


class TestSessionStoreTtl:
    """Tests for session TTL configuration and expires_at computation."""

    def test_default_ttl_is_24(self):
        store = SessionStore()
        assert store._session_ttl_hours == 24

    def test_custom_ttl(self):
        store = SessionStore(session_ttl_hours=48)
        assert store._session_ttl_hours == 48

    async def test_persist_upserts_with_expires_at(self):
        """_persist_to_db passes expires_at based on TTL when upserting."""
        from datetime import UTC, datetime, timedelta

        mock_db = MagicMock()
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _ctx():
            yield mock_async_session

        mock_db.session_context = _ctx

        store = SessionStore(database_service=mock_db, session_ttl_hours=12)
        ctx = store.get_context("s1")
        ctx["title"] = "TTL test"

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)

            before = datetime.now(UTC)
            await store._persist_to_db("s1")
            after = datetime.now(UTC)
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        upsert_kwargs = mock_repo.upsert.call_args[1]
        expires_at = upsert_kwargs["expires_at"]
        expected_min = before + timedelta(hours=12)
        expected_max = after + timedelta(hours=12)
        assert expected_min <= expires_at <= expected_max


class TestSessionStoreEviction:
    """Tests for in-memory session eviction."""

    def test_evict_if_needed_does_nothing_under_threshold(self):
        store = SessionStore()
        for i in range(10):
            store.get_context(f"s{i}")
        store._evict_if_needed()
        assert store.session_count() == 10

    def test_evict_if_needed_evicts_at_threshold(self):
        store = SessionStore()
        # Exceed threshold by 1
        for i in range(_MAX_MEMORY_SESSIONS + 1):
            store.get_context(f"s{i}")
        assert store.session_count() == _MAX_MEMORY_SESSIONS + 1

        store._evict_if_needed()

        # Should have evicted half
        expected = _MAX_MEMORY_SESSIONS + 1 - (_MAX_MEMORY_SESSIONS + 1) // 2
        assert store.session_count() == expected

    def test_evict_removes_oldest_sessions(self):
        store = SessionStore()
        # Create sessions in order
        for i in range(_MAX_MEMORY_SESSIONS + 10):
            store.get_context(f"s{i}")

        store._evict_if_needed()

        # Oldest sessions (lowest indices) should be evicted
        assert store.has_session("s0") is False
        assert store.has_session("s1") is False
        # Newest sessions should remain
        assert store.has_session(f"s{_MAX_MEMORY_SESSIONS + 9}") is True

    async def test_get_context_async_triggers_eviction_on_cache_miss(self):
        store = SessionStore()
        # Fill to over threshold
        for i in range(_MAX_MEMORY_SESSIONS + 1):
            store.get_context(f"s{i}")

        # Cache miss triggers eviction
        await store.get_context_async("new-session")

        # Should have evicted + added new one
        assert store.has_session("new-session") is True
        assert store.session_count() <= _MAX_MEMORY_SESSIONS + 1


class TestSessionStoreCleanupExpired:
    """Tests for cleanup_expired delegation."""

    async def test_cleanup_expired_returns_zero_without_db(self):
        store = SessionStore()
        result = await store.cleanup_expired()
        assert result == 0

    async def test_cleanup_expired_delegates_to_repo(self):
        mock_db = MagicMock()
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _ctx():
            yield mock_async_session

        mock_db.session_context = _ctx

        store = SessionStore(database_service=mock_db)

        mock_repo = MagicMock()
        mock_repo.cleanup_expired = AsyncMock(return_value=5)

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)
            result = await store.cleanup_expired()
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        assert result == 5
        mock_repo.cleanup_expired.assert_awaited_once()

    async def test_cleanup_expired_propagates_db_error(self):
        mock_db = MagicMock()

        @asynccontextmanager
        async def _failing_ctx():
            raise RuntimeError("DB down")
            yield  # noqa: F401

        mock_db.session_context = _failing_ctx

        store = SessionStore(database_service=mock_db)

        with pytest.raises(RuntimeError, match="DB down"):
            await store.cleanup_expired()


class TestPendingToolCallExpiry:
    """Tests for pending tool call expiry and rejected call ID storage."""

    def test_set_pending_stores_created_at(self):
        """set_pending_tool_call stores a float timestamp."""
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-1", "navigate")
        ctx = store.get_context("s1")
        created_at = ctx["pending_tool_call"]["created_at"]
        assert isinstance(created_at, float)

    def test_set_pending_stores_rejected_call_ids(self):
        """Rejected call IDs are stored when provided."""
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-1", "navigate", rejected_call_ids=["tc-2", "tc-3"])
        ctx = store.get_context("s1")
        assert ctx["pending_tool_call"]["rejected_call_ids"] == ["tc-2", "tc-3"]

    def test_set_pending_no_rejected_ids_omits_key(self):
        """When no rejected IDs, the key is absent from the stored dict."""
        store = SessionStore()
        store.set_pending_tool_call("s1", "tc-1", "navigate")
        ctx = store.get_context("s1")
        assert "rejected_call_ids" not in ctx["pending_tool_call"]

    def test_clear_pending_returns_none_when_expired(self):
        """Expired pending tool call returns None."""
        store = SessionStore(pending_tool_call_timeout_minutes=5)
        store.set_pending_tool_call("s1", "tc-1", "navigate")

        # Simulate 6 minutes elapsed
        with patch("lovely_assistant.app.assistant._session_store.time") as mock_time:
            # set_pending used real time.time(), so we need to override the clear check
            created_at = store.get_context("s1")["pending_tool_call"]["created_at"]
            mock_time.time.return_value = created_at + (6 * 60)  # 6 minutes later
            result = store.clear_pending_tool_call("s1")

        assert result is None

    def test_clear_pending_returns_valid_when_not_expired(self):
        """Non-expired pending tool call returns the stored dict."""
        store = SessionStore(pending_tool_call_timeout_minutes=10)
        store.set_pending_tool_call("s1", "tc-1", "navigate")

        # Simulate 3 minutes elapsed
        with patch("lovely_assistant.app.assistant._session_store.time") as mock_time:
            created_at = store.get_context("s1")["pending_tool_call"]["created_at"]
            mock_time.time.return_value = created_at + (3 * 60)  # 3 minutes later
            result = store.clear_pending_tool_call("s1")

        assert result is not None
        assert result["tool_call_id"] == "tc-1"

    def test_clear_pending_no_created_at_backward_compat(self):
        """Old dict without created_at still returns (backward compat)."""
        store = SessionStore()
        # Manually set an old-format pending tool call (no created_at)
        ctx = store.get_context("s1")
        ctx["pending_tool_call"] = {
            "tool_call_id": "tc-old",
            "tool_name": "navigate",
        }

        result = store.clear_pending_tool_call("s1")
        assert result is not None
        assert result["tool_call_id"] == "tc-old"


class TestSessionStoreDbRetry:
    """Tests for retry behavior on transient DB errors in _persist_to_db and _load_session_from_db."""

    async def test_persist_retries_on_operational_error_then_succeeds(self):
        """_persist_to_db retries on OperationalError and succeeds."""
        from sqlalchemy.exc import OperationalError

        mock_db = _make_mock_db()
        call_count = 0
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _flaky_context():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OperationalError("connection lost", None, None)
            yield mock_async_session

        mock_db.session_context = _flaky_context

        store = SessionStore(database_service=mock_db)
        ctx = store.get_context("s1")
        ctx["title"] = "Retry test"

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)
            await store._persist_to_db("s1")
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        # Should have called session_context twice (1 fail + 1 success)
        assert call_count == 2
        mock_repo.upsert.assert_awaited_once()

    async def test_persist_gives_up_after_max_retries(self):
        """_persist_to_db logs warning after exhausting retries."""
        from sqlalchemy.exc import OperationalError

        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _always_fail():
            raise OperationalError("DB down permanently", None, None)
            yield  # noqa: F401

        mock_db.session_context = _always_fail

        store = SessionStore(database_service=mock_db)
        store.get_context("s1")

        # Should not raise — _persist_to_db catches the exception after retries
        await store._persist_to_db("s1")

    async def test_load_retries_on_disconnection_error_then_succeeds(self):
        """_load_session_from_db retries on DisconnectionError and succeeds."""
        from sqlalchemy.exc import DisconnectionError

        mock_db = _make_mock_db()
        call_count = 0
        mock_async_session = AsyncMock()

        @asynccontextmanager
        async def _flaky_context():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise DisconnectionError("connection dropped")
            yield mock_async_session

        mock_db.session_context = _flaky_context

        store = SessionStore(database_service=mock_db)

        mock_row = MagicMock()
        mock_row.turn_number = 5
        mock_row.working_memory = None
        mock_row.pending_tool_call = None
        mock_row.message_history = []
        mock_row.title = "Loaded session"

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_row)

        import lovely_assistant.services.database.repositories as repo_mod

        original_class = getattr(repo_mod, "SessionRepository", None)
        try:
            repo_mod.SessionRepository = MagicMock(return_value=mock_repo)
            result = await store._load_session_from_db("s1")
        finally:
            if original_class is not None:
                repo_mod.SessionRepository = original_class

        assert result is not None
        assert result["turn_number"] == 5
        assert call_count == 2

    async def test_load_gives_up_returns_none(self):
        """_load_session_from_db returns None after exhausting retries."""
        from sqlalchemy.exc import OperationalError

        mock_db = _make_mock_db()

        @asynccontextmanager
        async def _always_fail():
            raise OperationalError("DB gone", None, None)
            yield  # noqa: F401

        mock_db.session_context = _always_fail

        store = SessionStore(database_service=mock_db)
        result = await store._load_session_from_db("s1")
        assert result is None
