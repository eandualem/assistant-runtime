from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from assistant_runtime.services.database.models import (
    MessageORM,
    SessionORM,
    SteeringORM,
    UserSettingsORM,
)
from assistant_runtime.services.database.repositories import (
    MessageRepository,
    SessionRepository,
    SettingsRepository,
    SteeringRepository,
)


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.execute = AsyncMock()
    session.flush = AsyncMock()
    return session


class TestSessionRepository:
    async def test_create_initializes_turn_number(self, mock_session: AsyncMock) -> None:
        row = SessionORM(id="sess-1", title=None, turn_number=0)
        result = MagicMock()
        result.scalar_one.return_value = row
        mock_session.execute.return_value = result

        created = await SessionRepository(mock_session).create("sess-1")

        assert created.id == "sess-1"
        assert created.turn_number == 0
        mock_session.flush.assert_awaited_once()

    async def test_upsert_accepts_tree_session_metadata(self, mock_session: AsyncMock) -> None:
        repo = SessionRepository(mock_session)

        await repo.upsert(
            "sess-1",
            title="Hello",
            turn_number=3,
            working_memory={"goal": "ship tree model"},
            telegram_chat_id="123456789",
            pending_action={"tool_call_id": "call-1", "tool_name": "navigate"},
        )

        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()
        statement = mock_session.execute.await_args.args[0]
        compiled = statement.compile(dialect=postgresql.dialect())
        assert compiled.params["pending_action"] == {
            "tool_call_id": "call-1",
            "tool_name": "navigate",
        }
        assert "pending_action = excluded.pending_action" in str(compiled)


class TestMessageRepository:
    async def test_create_persists_tree_message_fields(self, mock_session: AsyncMock) -> None:
        row = MessageORM(
            id="assistant-1",
            session_id="sess-1",
            parent_id="user-1",
            role="assistant",
            message_type="standard",
            content="Hello",
        )
        result = MagicMock()
        result.scalar_one.return_value = row
        mock_session.execute.return_value = result

        created = await MessageRepository(mock_session).create(
            message_id="assistant-1",
            session_id="sess-1",
            parent_id="user-1",
            role="assistant",
            message_type="standard",
            content="Hello",
            segments=[{"kind": "text", "text": "Hello"}],
            usage={"input_tokens": 1, "output_tokens": 2},
        )

        assert created.id == "assistant-1"
        assert created.parent_id == "user-1"
        mock_session.flush.assert_awaited_once()

    async def test_count_by_sessions_returns_mapping(self, mock_session: AsyncMock) -> None:
        result = MagicMock()
        result.all.return_value = [("sess-1", 2), ("sess-2", 5)]
        mock_session.execute.return_value = result

        counts = await MessageRepository(mock_session).count_by_sessions(["sess-1", "sess-2"])

        assert counts == {"sess-1": 2, "sess-2": 5}

    async def test_update_is_noop_when_no_fields_provided(self, mock_session: AsyncMock) -> None:
        await MessageRepository(mock_session).update("assistant-1")

        mock_session.execute.assert_not_called()
        mock_session.flush.assert_not_called()


class TestSteeringRepository:
    @pytest.mark.parametrize("profile", [None, "editor"])
    async def test_create_persists_steering_fields(self, mock_session: AsyncMock, profile) -> None:
        row = SteeringORM(
            id="steering-1",
            session_id="sess-1",
            content="Focus on Planner",
            status="pending",
        )
        result = MagicMock()
        result.scalar_one.return_value = row
        mock_session.execute.return_value = result

        created = await SteeringRepository(mock_session).create(
            steering_id="steering-1",
            session_id="sess-1",
            content="Focus on Planner",
            status="pending",
            profile=profile,
        )

        assert created.id == "steering-1"
        assert created.status == "pending"
        mock_session.flush.assert_awaited_once()
        statement = mock_session.execute.await_args.args[0]
        assert statement.compile(dialect=postgresql.dialect()).params["profile"] == profile

    async def test_mark_status_is_noop_for_empty_id_list(self, mock_session: AsyncMock) -> None:
        await SteeringRepository(mock_session).mark_status(
            [], status="delivered", delivered_at=None
        )

        mock_session.execute.assert_not_called()
        mock_session.flush.assert_not_called()


class TestSettingsRepository:
    async def test_save_returns_existing_row_when_state_is_empty(
        self, mock_session: AsyncMock
    ) -> None:
        row = UserSettingsORM(id="default")
        result = MagicMock()
        result.scalar_one_or_none.return_value = row
        mock_session.execute.return_value = result

        saved = await SettingsRepository(mock_session).save({})

        assert saved is row


class TestOAuthTokenRepository:
    async def test_each_statement_touches_only_its_own_kind(self, mock_session: AsyncMock) -> None:
        from assistant_runtime.services.database.repositories import OAuthTokenRepository

        repo = OAuthTokenRepository(mock_session)
        mock_session.execute.return_value = MagicMock(rowcount=0)
        await repo.upsert("openai", "api_key", encrypted_api_key="stored-key")
        await repo.delete("openai", "login")
        await repo.get("openai", "login")

        upsert, delete, get = (
            str(call.args[0].compile(dialect=postgresql.dialect()))
            for call in mock_session.execute.await_args_list
        )
        # The kind is part of the key, so a stored key and a login never share a row.
        assert "ON CONFLICT (provider, kind) DO UPDATE" in upsert
        assert "oauth_tokens.kind = " in delete
        assert "oauth_tokens.kind = " in get


class TestReadApiQueries:
    """The keyset, ``updated_after`` and count filters reach the SQL."""

    async def test_filters_and_counts_compile(self, mock_session: AsyncMock) -> None:
        from datetime import UTC, datetime

        from assistant_runtime.services.database.repositories import (
            ActionRepository,
            AgentRepository,
            EventRepository,
            TaskRepository,
        )

        when = datetime(2026, 10, 1, tzinfo=UTC)
        mock_session.execute.return_value = MagicMock(scalar_one=MagicMock(return_value=0))
        await SessionRepository(mock_session).list_all(updated_after=when)
        await TaskRepository(mock_session).list(before=(when, "t1"))
        await AgentRepository(mock_session).list_messages(before=(when, "m1"))
        await ActionRepository(mock_session).list(limit=10, before=7)
        await MessageRepository(mock_session).count_created(when, when)

        sessions, tasks, messages, actions, count = (
            str(call.args[0].compile(dialect=postgresql.dialect()))
            for call in mock_session.execute.await_args_list
        )
        assert "sessions.updated_at > " in sessions
        assert "(tasks.created_at, tasks.id) < (" in tasks
        assert "(agent_messages.created_at, agent_messages.id) < (" in messages
        assert "actions.id < " in actions
        assert "count(*)" in count
        assert "messages.created_at < " in count
        assert await EventRepository(mock_session).count_created(when, when) == 0


class TestChangeCursors:
    """Every write sets ``updated_at``; change lists filter and order by ``(updated_at, id)``."""

    async def test_every_update_sets_updated_at(self, mock_session: AsyncMock) -> None:
        from assistant_runtime.services.database.repositories import (
            ActionRepository,
            AgentRepository,
            EventRepository,
            TaskRepository,
        )

        mock_session.execute.return_value = MagicMock()
        tasks, agents = TaskRepository(mock_session), AgentRepository(mock_session)
        await tasks.update("t1", only_from=("queued",), status="running")
        await tasks.mark_unfinished("interrupted", "restart")
        await agents.update_message("m1", status="done")
        await agents.mark_unfinished_messages("interrupted", "restart")
        await ActionRepository(mock_session).update(1, status="sent")
        await EventRepository(mock_session).update(1, status="heard")
        await MessageRepository(mock_session).update("a1", content="Edited")

        for call in mock_session.execute.await_args_list:
            sql = str(call.args[0].compile(dialect=postgresql.dialect()))
            assert sql.startswith("UPDATE ")
            assert "updated_at=now()" in sql

    async def test_change_lists_compile(self, mock_session: AsyncMock) -> None:
        from datetime import UTC, datetime

        from assistant_runtime.base.cursors import ChangeCursor
        from assistant_runtime.services.database.repositories import (
            ActionRepository,
            AgentRepository,
            EventRepository,
            TaskRepository,
        )

        mock_session.execute.return_value = MagicMock()
        at = ChangeCursor(datetime(2026, 10, 1, tzinfo=UTC), "x")
        bare = ChangeCursor(at.updated_at)
        await MessageRepository(mock_session).list_changes("s1", after=at, limit=5)
        await MessageRepository(mock_session).list_changes("s1", after=None, limit=5)
        await TaskRepository(mock_session).list(updated_after=bare)
        await AgentRepository(mock_session).list_messages(agent_id="a1", updated_after=at)
        await ActionRepository(mock_session).list(limit=5, updated_after=bare, status="sent")
        await EventRepository(mock_session).list(after=0, limit=5, updated_after=at)

        messages, everything, tasks, agent_messages, actions, events = (
            str(call.args[0].compile(dialect=postgresql.dialect()))
            for call in mock_session.execute.await_args_list
        )
        assert "messages.session_id = " in messages
        assert "(messages.updated_at, messages.id) > (" in messages
        assert "updated_at >" not in everything  # no cursor: from the beginning
        assert "tasks.updated_at > " in tasks
        assert "(agent_messages.updated_at, agent_messages.id) > (" in agent_messages
        assert "actions.status = " in actions  # other filters still apply
        assert "(events.updated_at, events.id) > (" in events
        for sql, table in (
            (messages, "messages"),
            (everything, "messages"),
            (tasks, "tasks"),
            (agent_messages, "agent_messages"),
            (actions, "actions"),
            (events, "events"),
        ):
            assert f"ORDER BY {table}.updated_at ASC, {table}.id ASC" in sql
