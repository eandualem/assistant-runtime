"""Tests for the in-memory session store."""

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
