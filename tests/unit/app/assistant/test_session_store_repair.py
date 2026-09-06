"""Tests for stale host tool repair functionality in SessionStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.assistant._session_persistence import (
    LoadedSession,
    pending_action_from_context,
)
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant._stale_tools import (
    STALE_HOST_TOOL_OUTPUT,
    find_tool_entry,
    repair_stale_tool_segments,
    repair_stale_tools_in_context,
)
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import create_app


def _request(
    *,
    message_id: str,
    session_id: str = "sess-1",
    parent_id: str | None = None,
    content: str = "Hello",
    message_type: str = "standard",
) -> AssistantRequest:
    return AssistantRequest(
        id=message_id,
        session_id=session_id,
        parent_id=parent_id,
        content=content,
        message_type=message_type,
    )


async def _seed_basic_turn(store: SessionStore) -> tuple[dict, dict, dict]:
    ctx, user = await store.register_user_message(_request(message_id="user-1", parent_id=None))
    assistant = await store.register_assistant_message(
        "sess-1",
        message_id="assistant-1",
        parent_id="user-1",
        content="Hi there",
        segments=[{"kind": "text", "text": "Hi there"}],
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    return ctx, user, assistant


def _create_test_app(*, sessions: SessionStore | None = None) -> Any:
    from assistant_runtime.base.lifecycle import LifecycleManager

    app = create_app()
    app.state.lifecycle = LifecycleManager()
    mock_service = MagicMock()
    mock_service.get_session_store.return_value = sessions or SessionStore()
    mock_service.get_database_service.return_value = None
    app.state.assistant_service = mock_service
    app.state.streaming_service = MagicMock()
    return app


# ---------------------------------------------------------------------------
# repair_stale_tool_segments
# ---------------------------------------------------------------------------


class TestRepairStaleToolSegments:
    def test_tool_without_output_gets_stale_output(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"id": "tool-1", "name": "ui_navigate"}],
            }
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == ["tool-1"]
        assert repaired[0]["tools"][0]["output"] == STALE_HOST_TOOL_OUTPUT

    def test_tool_with_output_is_unchanged(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"id": "tool-1", "name": "ui_navigate", "output": "done"}],
            }
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == []
        assert repaired[0]["tools"][0]["output"] == "done"

    def test_mixed_tools_only_repairs_missing(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [
                    {"id": "tool-1", "name": "ui_navigate", "output": "ok"},
                    {"id": "tool-2", "name": "ui_send_event"},
                    {"id": "tool-3", "name": "ui_navigate"},
                ],
            }
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == ["tool-2", "tool-3"]
        assert repaired[0]["tools"][0]["output"] == "ok"
        assert repaired[0]["tools"][1]["output"] == STALE_HOST_TOOL_OUTPUT
        assert repaired[0]["tools"][2]["output"] == STALE_HOST_TOOL_OUTPUT

    def test_non_dict_segment_is_skipped(self) -> None:
        segments = ["garbage", {"kind": "tool_group", "tools": [{"id": "t1", "name": "x"}]}]
        repaired, ids = repair_stale_tool_segments(segments)
        assert ids == ["t1"]
        assert repaired[0] == "garbage"

    def test_empty_segments_returns_empty(self) -> None:
        repaired, ids = repair_stale_tool_segments([])

        assert repaired == []
        assert ids == []

    def test_non_tool_segments_untouched(self) -> None:
        segments = [
            {"kind": "text", "text": "Hello world"},
            {"kind": "thinking", "text": "Let me think..."},
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == []
        assert repaired == segments

    def test_original_segments_not_mutated(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"id": "tool-1", "name": "ui_navigate"}],
            }
        ]

        repair_stale_tool_segments(segments)

        assert "output" not in segments[0]["tools"][0]

    def test_multiple_tool_groups(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"id": "tool-1", "name": "ui_navigate", "output": "ok"}],
            },
            {"kind": "text", "text": "Between groups"},
            {
                "kind": "tool_group",
                "tools": [{"id": "tool-2", "name": "ui_send_event"}],
            },
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == ["tool-2"]
        assert repaired[0]["tools"][0]["output"] == "ok"
        assert repaired[2]["tools"][0]["output"] == STALE_HOST_TOOL_OUTPUT

    def test_tool_without_id_uses_empty_string(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"name": "ui_navigate"}],
            }
        ]

        repaired, ids = repair_stale_tool_segments(segments)

        assert ids == [""]


# ---------------------------------------------------------------------------
# repair_stale_tools_in_context
# ---------------------------------------------------------------------------


class TestRepairStaleToolsInContext:
    def test_context_with_stale_assistant_tool_is_repaired(self) -> None:
        ctx: dict[str, Any] = {
            "message_index": {
                "user-1": {"id": "user-1", "role": "user", "segments": None},
                "assistant-1": {
                    "id": "assistant-1",
                    "role": "assistant",
                    "segments": [
                        {
                            "kind": "tool_group",
                            "tools": [{"id": "tool-1", "name": "ui_navigate"}],
                        }
                    ],
                },
            }
        }

        repaired = repair_stale_tools_in_context(ctx)

        assert len(repaired) == 1
        msg_id, segments = repaired[0]
        assert msg_id == "assistant-1"
        assert segments[0]["tools"][0]["output"] == STALE_HOST_TOOL_OUTPUT
        # In-place mutation on ctx
        assert (
            ctx["message_index"]["assistant-1"]["segments"][0]["tools"][0]["output"]
            == STALE_HOST_TOOL_OUTPUT
        )

    def test_user_only_messages_no_repairs(self) -> None:
        ctx: dict[str, Any] = {
            "message_index": {
                "user-1": {"id": "user-1", "role": "user", "segments": None},
                "user-2": {"id": "user-2", "role": "user", "segments": None},
            }
        }

        repaired = repair_stale_tools_in_context(ctx)

        assert repaired == []

    def test_all_tools_have_output_no_repairs(self) -> None:
        ctx: dict[str, Any] = {
            "message_index": {
                "assistant-1": {
                    "id": "assistant-1",
                    "role": "assistant",
                    "segments": [
                        {
                            "kind": "tool_group",
                            "tools": [{"id": "tool-1", "name": "ui_navigate", "output": "done"}],
                        }
                    ],
                },
            }
        }

        repaired = repair_stale_tools_in_context(ctx)

        assert repaired == []

    def test_multiple_assistants_only_stale_one_repaired(self) -> None:
        ctx: dict[str, Any] = {
            "message_index": {
                "assistant-1": {
                    "id": "assistant-1",
                    "role": "assistant",
                    "segments": [
                        {
                            "kind": "tool_group",
                            "tools": [{"id": "tool-1", "name": "ui_navigate", "output": "ok"}],
                        }
                    ],
                },
                "assistant-2": {
                    "id": "assistant-2",
                    "role": "assistant",
                    "segments": [
                        {
                            "kind": "tool_group",
                            "tools": [{"id": "tool-2", "name": "ui_send_event"}],
                        }
                    ],
                },
            }
        }

        repaired = repair_stale_tools_in_context(ctx)

        assert len(repaired) == 1
        assert repaired[0][0] == "assistant-2"

    def test_assistant_without_segments_skipped(self) -> None:
        ctx: dict[str, Any] = {
            "message_index": {
                "assistant-1": {
                    "id": "assistant-1",
                    "role": "assistant",
                    "segments": None,
                },
                "assistant-2": {
                    "id": "assistant-2",
                    "role": "assistant",
                    "segments": [],
                },
            }
        }

        repaired = repair_stale_tools_in_context(ctx)

        assert repaired == []


# ---------------------------------------------------------------------------
# SessionStore.repair_stale_host_tools
# ---------------------------------------------------------------------------


class TestRepairStaleHostTools:
    async def test_clears_pending_tool_call_state(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)
        ctx = store.get_context("sess-1")
        ctx["pending_tool_call_id"] = "call-123"
        ctx["pending_tool_name"] = "ui_navigate"
        ctx["pending_assistant_message_id"] = "assistant-1"

        report = await store.repair_stale_host_tools("sess-1")

        assert report["cleared_pending"]["tool_call_id"] == "call-123"
        assert report["cleared_pending"]["tool_name"] == "ui_navigate"
        assert ctx.get("pending_tool_call_id") is None
        assert ctx.get("pending_tool_name") is None
        assert ctx.get("pending_assistant_message_id") is None

    async def test_repairs_stale_tool_segments(self) -> None:
        store = SessionStore()
        await store.register_user_message(_request(message_id="user-1", parent_id=None))
        await store.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Navigating...",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [{"id": "tool-1", "name": "ui_navigate"}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2},
        )

        report = await store.repair_stale_host_tools("sess-1")

        assert "assistant-1" in report["repaired_tools"]
        ctx = store.get_context("sess-1")
        assistant_record = ctx["message_index"]["assistant-1"]
        assert assistant_record["segments"][0]["tools"][0]["output"] == STALE_HOST_TOOL_OUTPUT

    async def test_repairs_both_pending_state_and_stale_segments(self) -> None:
        store = SessionStore()
        await store.register_user_message(_request(message_id="user-1", parent_id=None))
        await store.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Running tool...",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [{"id": "tool-1", "name": "ui_navigate"}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        ctx = store.get_context("sess-1")
        ctx["pending_tool_call_id"] = "call-456"
        ctx["pending_tool_name"] = "ui_navigate"
        ctx["pending_assistant_message_id"] = "assistant-1"

        report = await store.repair_stale_host_tools("sess-1")

        assert report["cleared_pending"] is not None
        assert report["cleared_pending"]["tool_call_id"] == "call-456"
        assert "assistant-1" in report["repaired_tools"]

    async def test_nothing_to_repair_returns_empty_report(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)

        report = await store.repair_stale_host_tools("sess-1")

        assert report["session_id"] == "sess-1"
        assert report["cleared_pending"] is None
        assert report["repaired_tools"] == []

    async def test_nonexistent_session_raises_lookup_error(self) -> None:
        store = SessionStore()

        with pytest.raises(LookupError, match="not found"):
            await store.repair_stale_host_tools("nonexistent")

    async def test_cached_path_updated_after_segment_repair(self) -> None:
        store = SessionStore()
        await store.register_user_message(_request(message_id="user-1", parent_id=None))
        await store.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Navigating...",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [{"id": "tool-1", "name": "ui_navigate"}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2},
        )

        await store.repair_stale_host_tools("sess-1")

        ctx = store.get_context("sess-1")
        # The cached_path should also reflect the repaired segments
        assistant_in_path = [m for m in ctx["cached_path"] if m["id"] == "assistant-1"]
        assert len(assistant_in_path) == 1
        assert assistant_in_path[0]["segments"][0]["tools"][0]["output"] == STALE_HOST_TOOL_OUTPUT


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/repair endpoint
# ---------------------------------------------------------------------------


class TestRepairEndpoint:
    @pytest.mark.asyncio
    async def test_repair_returns_report(self) -> None:
        sessions = SessionStore()
        await sessions.register_user_message(_request(message_id="user-1", parent_id=None))
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Navigating...",
            segments=[
                {
                    "kind": "tool_group",
                    "tools": [{"id": "tool-1", "name": "ui_navigate"}],
                }
            ],
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/sessions/sess-1/repair")

        assert response.status_code == 200
        body = response.json()
        assert body["session_id"] == "sess-1"
        assert "assistant-1" in body["repaired_tools"]

    @pytest.mark.asyncio
    async def test_repair_nonexistent_session_returns_404(self) -> None:
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/sessions/nonexistent/repair")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_repair_clean_session_returns_empty_report(self) -> None:
        sessions = SessionStore()
        await sessions.register_user_message(_request(message_id="user-1", parent_id=None))
        await sessions.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="All good",
            segments=[{"kind": "text", "text": "All good"}],
            usage={"input_tokens": 1, "output_tokens": 2},
        )
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/sessions/sess-1/repair")

        assert response.status_code == 200
        body = response.json()
        assert body["cleared_pending"] is None
        assert body["repaired_tools"] == []


# ---------------------------------------------------------------------------
# Pending host action: persisted on the row and restored on load
# ---------------------------------------------------------------------------


def _pending_segments(call_id: str = "call-1", *, answered: bool = False) -> list[dict[str, Any]]:
    tool: dict[str, Any] = {"id": call_id, "name": "ui_navigate", "input": {"page": "home"}}
    if answered:
        tool["output"] = {"ok": True}
    return [{"kind": "tool_group", "tools": [tool]}]


def _loaded(
    pending_action: dict[str, Any] | None, *, answered: bool = False, call_id: str = "call-1"
) -> LoadedSession:
    now = datetime.now(UTC)
    return LoadedSession(
        turn_number=1,
        working_memory=None,
        title="Hello",
        owner_id=None,
        telegram_chat_id=None,
        telegram_bound_at=None,
        messages=[
            {
                "id": "user-1",
                "session_id": "sess-1",
                "parent_id": None,
                "role": "user",
                "message_type": "standard",
                "content": "Hello",
                "segments": None,
                "usage": None,
                "created_at": now,
            },
            {
                "id": "assistant-1",
                "session_id": "sess-1",
                "parent_id": "user-1",
                "role": "assistant",
                "message_type": "standard",
                "content": "",
                "segments": _pending_segments(call_id, answered=answered),
                "usage": None,
                "created_at": now + timedelta(seconds=1),
            },
        ],
        steering=[],
        pending_action=pending_action,
    )


def _store_with_db(loaded: LoadedSession | None = None) -> tuple[SessionStore, AsyncMock]:
    store = SessionStore()
    db = AsyncMock()
    db.load.return_value = loaded
    store._db = db
    return store, db


class TestPendingActionPersistence:
    async def test_set_pending_action_writes_the_row(self) -> None:
        store, db = _store_with_db()
        await _seed_basic_turn(store)
        db.save_state.reset_mock()

        await store.set_pending_action(
            "sess-1", tool_call_id="call-1", tool_name="ui_navigate", assistant_message_id="a-1"
        )

        ctx = store.get_context("sess-1")
        assert ctx["pending_tool_call_id"] == "call-1"
        assert ctx["pending_tool_name"] == "ui_navigate"
        assert ctx["pending_assistant_message_id"] == "a-1"
        db.save_state.assert_awaited_once()
        saved_ctx = db.save_state.await_args.args[1]
        assert pending_action_from_context(saved_ctx) == {
            "tool_call_id": "call-1",
            "tool_name": "ui_navigate",
            "assistant_message_id": "a-1",
            "batch": ["call-1"],
        }

    async def test_clear_pending_action_writes_the_row_only_when_something_was_pending(
        self,
    ) -> None:
        store, db = _store_with_db()
        await _seed_basic_turn(store)
        db.save_state.reset_mock()

        assert await store.clear_pending_action("sess-1") is None
        db.save_state.assert_not_awaited()

        await store.set_pending_action(
            "sess-1", tool_call_id="call-1", tool_name="ui_navigate", assistant_message_id="a-1"
        )
        cleared = await store.clear_pending_action("sess-1")

        assert cleared == {
            "tool_call_id": "call-1",
            "tool_name": "ui_navigate",
            "assistant_message_id": "a-1",
            "batch": ["call-1"],
        }
        assert db.save_state.await_count == 2
        assert pending_action_from_context(db.save_state.await_args.args[1]) is None
        assert store.get_context("sess-1").get("pending_tool_call_id") is None

    async def test_load_restores_a_stored_pending_action_and_repairs_nothing(self) -> None:
        pending = {
            "tool_call_id": "call-1",
            "tool_name": "ui_navigate",
            "assistant_message_id": "assistant-1",
        }
        store, db = _store_with_db(_loaded(pending))

        ctx = await store.get_context_if_exists_async("sess-1")

        assert ctx["pending_tool_call_id"] == "call-1"
        assert ctx["pending_tool_name"] == "ui_navigate"
        assert ctx["pending_assistant_message_id"] == "assistant-1"
        assert "output" not in ctx["message_index"]["assistant-1"]["segments"][0]["tools"][0]
        db.update_segments.assert_not_awaited()
        db.save_state.assert_not_awaited()

    async def test_load_restores_the_whole_batch_unanswered(self) -> None:
        loaded = _loaded(
            {
                "tool_call_id": "call-1",
                "tool_name": "ui_navigate",
                "assistant_message_id": "assistant-1",
                "batch": ["call-1", "call-2", "gone"],
            }
        )
        loaded.messages[1]["segments"][0]["tools"].append(
            {"id": "call-2", "name": "ui_navigate", "input": {"page": "b"}}
        )
        store, db = _store_with_db(loaded)

        ctx = await store.get_context_if_exists_async("sess-1")

        assert ctx["pending_tool_call_id"] == "call-1"
        assert ctx["pending_tool_batch"] == ["call-1", "call-2"]  # unknown ids dropped
        tools = ctx["message_index"]["assistant-1"]["segments"][0]["tools"]
        assert all("output" not in tool for tool in tools)  # the queued call is kept open
        db.update_segments.assert_not_awaited()

    async def test_load_without_a_stored_pending_action_marks_the_call_unknown(self) -> None:
        store, db = _store_with_db(_loaded(None))

        ctx = await store.get_context_if_exists_async("sess-1")

        assert ctx["pending_tool_call_id"] is None
        tool = ctx["message_index"]["assistant-1"]["segments"][0]["tools"][0]
        assert tool["output"] == STALE_HOST_TOOL_OUTPUT
        assert tool["outcome"] == "interrupted"
        assert tool["status"] == "unknown"
        db.update_segments.assert_awaited_once()

    @pytest.mark.parametrize(
        "pending",
        [
            {"tool_call_id": "call-1", "tool_name": "ui_navigate", "assistant_message_id": "gone"},
            {
                "tool_call_id": "other",
                "tool_name": "ui_navigate",
                "assistant_message_id": "assistant-1",
            },
            {"tool_call_id": None, "tool_name": None, "assistant_message_id": None},
        ],
        ids=["unknown-message", "unknown-call", "empty"],
    )
    async def test_load_drops_a_stored_pending_action_that_does_not_match(self, pending) -> None:
        store, db = _store_with_db(_loaded(pending))

        ctx = await store.get_context_if_exists_async("sess-1")

        assert ctx["pending_tool_call_id"] is None
        # The mismatching call is unanswered, so it is resolved as unknown and the row cleared.
        assert ctx["message_index"]["assistant-1"]["segments"][0]["tools"][0]["status"] == "unknown"
        db.save_state.assert_awaited_once()
        assert pending_action_from_context(db.save_state.await_args.args[1]) is None

    async def test_load_drops_a_stored_pending_action_whose_result_is_recorded(self) -> None:
        pending = {
            "tool_call_id": "call-1",
            "tool_name": "ui_navigate",
            "assistant_message_id": "assistant-1",
        }
        store, db = _store_with_db(_loaded(pending, answered=True))

        ctx = await store.get_context_if_exists_async("sess-1")

        assert ctx["pending_tool_call_id"] is None
        assert ctx["message_index"]["assistant-1"]["segments"][0]["tools"][0]["output"] == {
            "ok": True
        }
        db.update_segments.assert_not_awaited()
        db.save_state.assert_awaited_once()

    async def test_repair_clears_the_row_and_records_unknown(self) -> None:
        pending = {
            "tool_call_id": "call-1",
            "tool_name": "ui_navigate",
            "assistant_message_id": "assistant-1",
        }
        store, db = _store_with_db(_loaded(pending))
        await store.get_context_if_exists_async("sess-1")

        report = await store.repair_stale_host_tools("sess-1")

        assert report["cleared_pending"] == {"tool_call_id": "call-1", "tool_name": "ui_navigate"}
        assert report["repaired_tools"] == ["assistant-1"]
        db.save_state.assert_awaited_once()
        db.update_segments.assert_awaited_once()
        tool = store.get_context("sess-1")["message_index"]["assistant-1"]["segments"][0]["tools"][
            0
        ]
        assert tool["status"] == "unknown"


class TestStaleToolHelpers:
    def test_keep_leaves_the_pending_call_unanswered(self) -> None:
        segments = [
            {
                "kind": "tool_group",
                "tools": [{"id": "keep-1", "name": "ui_navigate"}, {"id": "old-1", "name": "ui_x"}],
            }
        ]

        repaired, ids = repair_stale_tool_segments(segments, keep=frozenset({"keep-1"}))

        assert ids == ["old-1"]
        assert "output" not in repaired[0]["tools"][0]
        assert repaired[0]["tools"][1]["status"] == "unknown"
        assert repaired[0]["tools"][1]["outcome"] == "interrupted"

    def test_find_tool_entry(self) -> None:
        segments = [
            {"kind": "text", "text": "hi"},
            {"kind": "tool_group", "tools": [{"id": "a", "name": "x"}, {"id": "b", "name": "y"}]},
        ]

        assert find_tool_entry(segments, "b") == {"id": "b", "name": "y"}
        assert find_tool_entry(segments, "c") is None
        assert find_tool_entry(None, "a") is None
