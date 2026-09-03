from __future__ import annotations

import pytest

from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.models import AssistantRequest


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


class TestSessionStoreBasics:
    def test_get_context_initializes_tree_and_steering_state(self) -> None:
        ctx = SessionStore().get_context("sess-1")

        assert ctx["turn_number"] == 0
        assert ctx["message_count"] == 0
        assert ctx["cached_path"] == []
        assert ctx["active_leaf_id"] is None
        assert ctx["pending_steering_ids"] == []
        assert ctx["steering_index"] == {}
        assert ctx["steering_order"] == []


class TestRegisterUserMessage:
    async def test_registers_first_root_message(self) -> None:
        store = SessionStore()

        ctx, record = await store.register_user_message(
            _request(message_id="user-1", parent_id=None)
        )

        assert record["id"] == "user-1"
        assert record["role"] == "user"
        assert ctx["turn_number"] == 1
        assert ctx["message_count"] == 1
        assert ctx["active_leaf_id"] == "user-1"
        assert [message["id"] for message in ctx["cached_path"]] == ["user-1"]
        assert ctx["title"] == "Hello"

    async def test_rejects_duplicate_message_ids(self) -> None:
        store = SessionStore()
        await store.register_user_message(_request(message_id="user-1", parent_id=None))

        with pytest.raises(ValueError, match="already exists"):
            await store.register_user_message(
                _request(message_id="user-1", parent_id="user-1", content="Duplicate")
            )

    async def test_rebuilds_cached_path_when_branching_from_ancestor(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)

        ctx, branch = await store.register_user_message(
            _request(
                message_id="user-2",
                parent_id="user-1",
                content="Actually, inspect Leo only",
            )
        )

        assert branch["parent_id"] == "user-1"
        assert ctx["active_leaf_id"] == "user-2"
        assert [message["id"] for message in ctx["cached_path"]] == ["user-1", "user-2"]

    async def test_rejects_steering_for_tree_registration(self) -> None:
        store = SessionStore()

        with pytest.raises(ValueError, match="stored separately"):
            await store.register_user_message(
                _request(
                    message_id="steering-1",
                    content="Focus on Leo",
                    message_type="steering",
                )
            )


class TestAssistantMessages:
    async def test_registers_assistant_message_and_history_path(self) -> None:
        store = SessionStore()
        await store.register_user_message(_request(message_id="user-1", parent_id=None))

        record = await store.register_assistant_message(
            "sess-1",
            message_id="assistant-1",
            parent_id="user-1",
            content="Hi there",
            segments=[{"kind": "text", "text": "Hi there"}],
            usage={"input_tokens": 1, "output_tokens": 2},
        )

        assert record["role"] == "assistant"
        assert record["parent_id"] == "user-1"
        history = store.get_history("sess-1")
        assert len(history) == 2

    async def test_update_message_rewrites_cached_record(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)

        record = await store.update_message(
            "sess-1",
            "assistant-1",
            content="Updated",
            segments=[{"kind": "text", "text": "Updated"}],
            usage={"input_tokens": 3, "output_tokens": 4},
        )

        assert record["content"] == "Updated"
        path = await store.get_message_path("sess-1")
        assert path[-1]["content"] == "Updated"
        assert path[-1]["usage"] == {"input_tokens": 3, "output_tokens": 4}


class TestSteeringMessages:
    async def test_queues_steering_in_submission_order(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)

        await store.queue_steering(
            "sess-1",
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
        )
        await store.queue_steering(
            "sess-1",
            _request(
                message_id="steering-2",
                content="Skip Ada",
                message_type="steering",
            ),
        )

        queued = await store.list_pending_steering("sess-1")

        assert [message["id"] for message in queued] == ["steering-1", "steering-2"]
        assert store.get_context("sess-1")["pending_steering_ids"] == ["steering-1", "steering-2"]

    async def test_delivers_steering_without_mutating_tree(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)
        await store.queue_steering(
            "sess-1",
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
        )
        await store.queue_steering(
            "sess-1",
            _request(
                message_id="steering-2",
                content="Skip Ada",
                message_type="steering",
            ),
        )

        delivered = await store.deliver_pending_steering("sess-1")
        ctx = store.get_context("sess-1")
        path = await store.get_message_path("sess-1")
        display_steering = await store.get_display_steering("sess-1")

        assert [record["id"] for record in delivered] == ["steering-1", "steering-2"]
        assert all(record["status"] == "delivered" for record in delivered)
        assert ctx["pending_steering_ids"] == []
        assert ctx["active_leaf_id"] == "assistant-1"
        assert [message["id"] for message in path] == ["user-1", "assistant-1"]
        assert [record["id"] for record in display_steering] == ["steering-1", "steering-2"]

    async def test_marks_steering_promoted_for_idle_delivery(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)
        await store.queue_steering(
            "sess-1",
            _request(
                message_id="steering-1",
                content="Focus on Leo only",
                message_type="steering",
            ),
        )

        promoted = await store.mark_steering_promoted("sess-1", "steering-1")

        assert promoted["status"] == "promoted"
        assert promoted["delivered_at"] is not None
        assert store.get_context("sess-1")["pending_steering_ids"] == []


class TestMessagePathResolution:
    async def test_get_message_path_defaults_to_latest_leaf(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)
        await store.register_user_message(
            _request(
                message_id="user-2",
                parent_id="assistant-1",
                content="Show me Ada too",
            )
        )

        path = await store.get_message_path("sess-1")

        assert [message["id"] for message in path] == ["user-1", "assistant-1", "user-2"]

    async def test_get_message_path_accepts_specific_leaf_id(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)
        await store.register_user_message(
            _request(message_id="user-2", parent_id="assistant-1", content="branch one")
        )
        await store.register_user_message(
            _request(message_id="user-3", parent_id="user-1", content="branch two")
        )

        path = await store.get_message_path("sess-1", leaf_id="user-2")

        assert [message["id"] for message in path] == ["user-1", "assistant-1", "user-2"]


class TestListSessions:
    async def test_list_sessions_uses_tree_message_counts(self) -> None:
        store = SessionStore()
        await _seed_basic_turn(store)

        result = await store.list_sessions()

        assert result == [
            {
                "session_id": "sess-1",
                "title": "Hello",
                "turn_number": 1,
                "message_count": 2,
                "created_at": None,
            }
        ]
