from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

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
        )

        mock_session.execute.assert_awaited_once()
        mock_session.flush.assert_awaited_once()


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
    async def test_create_persists_steering_fields(self, mock_session: AsyncMock) -> None:
        row = SteeringORM(
            id="steering-1",
            session_id="sess-1",
            content="Focus on Leo",
            status="pending",
        )
        result = MagicMock()
        result.scalar_one.return_value = row
        mock_session.execute.return_value = result

        created = await SteeringRepository(mock_session).create(
            steering_id="steering-1",
            session_id="sess-1",
            content="Focus on Leo",
            status="pending",
        )

        assert created.id == "steering-1"
        assert created.status == "pending"
        mock_session.flush.assert_awaited_once()

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
