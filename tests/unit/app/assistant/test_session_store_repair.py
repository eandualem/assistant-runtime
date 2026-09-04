"""Tests for stale host tool repair functionality in SessionStore."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant._stale_tools import (
    STALE_HOST_TOOL_OUTPUT,
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
