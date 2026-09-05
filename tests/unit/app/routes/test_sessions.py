from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.assistant._session_store import SessionStore
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


async def _seed_branching_session(store: SessionStore) -> None:
    await store.register_user_message(_request(message_id="user-1", parent_id=None))
    await store.register_assistant_message(
        "sess-1",
        message_id="assistant-1",
        parent_id="user-1",
        content="Answer one",
        segments=[{"kind": "text", "text": "Answer one"}],
        usage={"input_tokens": 1, "output_tokens": 2},
    )
    await store.register_user_message(
        _request(message_id="user-2", parent_id="assistant-1", content="Branch one")
    )
    await store.register_user_message(
        _request(message_id="user-3", parent_id="user-1", content="Branch two")
    )


class TestListSessions:
    @pytest.mark.asyncio
    async def test_returns_tree_message_counts(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions")

        assert response.status_code == 200
        assert response.json() == [
            {
                "session_id": "sess-1",
                "owner_id": None,
                "title": "Hello",
                "turn_number": 3,
                "message_count": 4,
                "created_at": None,
            }
        ]


class TestGetSession:
    @pytest.mark.asyncio
    async def test_returns_session_metadata(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1")

        assert response.status_code == 200
        assert response.json() == {
            "session_id": "sess-1",
            "owner_id": None,
            "turn_number": 3,
            "has_pending_tool_call": False,
            "message_count": 4,
        }

    @pytest.mark.asyncio
    async def test_missing_session_returns_404(self) -> None:
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/missing")

        assert response.status_code == 404
        assert response.json()["detail"] == "Session not found"


class TestGetSessionMessages:
    @pytest.mark.asyncio
    async def test_returns_latest_leaf_by_default(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages")

        assert response.status_code == 200
        assert [message["id"] for message in response.json()] == ["user-1", "user-3"]

    @pytest.mark.asyncio
    async def test_leaf_id_selects_specific_branch(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages?leaf_id=user-2")

        assert response.status_code == 200
        assert [message["id"] for message in response.json()] == [
            "user-1",
            "assistant-1",
            "user-2",
        ]

    @pytest.mark.asyncio
    async def test_unknown_leaf_returns_404(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages?leaf_id=missing")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_messages_include_delivered_steering_without_tree_mutation(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        await sessions.queue_steering(
            "sess-1",
            _request(
                message_id="steering-1",
                content="Focus on Leo",
                message_type="steering",
            ),
        )
        await sessions.deliver_pending_steering("sess-1")
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/messages?leaf_id=assistant-1")

        assert response.status_code == 200
        messages = response.json()
        assert [message["id"] for message in messages] == ["user-1", "assistant-1", "steering-1"]
        assert messages[-1]["role"] == "steering"


class TestGetSessionTree:
    @pytest.mark.asyncio
    async def test_returns_all_messages_with_parent_links(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/sessions/sess-1/tree")

        assert response.status_code == 200
        tree = response.json()
        assert len(tree) == 4
        assert tree[0]["parent_id"] is None
        assert any(
            message["id"] == "assistant-1" and message["parent_id"] == "user-1" for message in tree
        )


class TestDeleteSession:
    @pytest.mark.asyncio
    async def test_deletes_existing_session(self) -> None:
        sessions = SessionStore()
        await _seed_branching_session(sessions)
        app = _create_test_app(sessions=sessions)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete("/api/sessions/sess-1")

        assert response.status_code == 200
        assert response.json() == {"session_id": "sess-1", "deleted": True}
        assert sessions.has_session("sess-1") is False

    @pytest.mark.asyncio
    async def test_deleted_branch_route_is_gone(self) -> None:
        app = _create_test_app()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/sessions/sess-1/branch", json={"message_id": "user-1"}
            )

        assert response.status_code == 404
