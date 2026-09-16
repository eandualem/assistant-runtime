"""Steering selector persistence and hydration, without a database server."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.services.database.models import SessionORM, SteeringORM
from assistant_runtime.services.database.repositories import (
    MessageRepository,
    SessionRepository,
    SteeringRepository,
)


async def test_profiles_survive_persistence_and_restart_in_submission_order():
    database = MagicMock()
    database.session_context.return_value = AsyncMock()
    rows = []

    async def insert(**values):
        values["id"] = values.pop("steering_id")
        row = SteeringORM(**values, created_at=datetime.now(UTC))
        rows.append(row)
        return row

    session = SessionORM(id="thread", turn_number=1, title="Conversation")
    with (
        patch.object(SteeringRepository, "create", AsyncMock(side_effect=insert)) as create,
        patch.object(SessionRepository, "get", AsyncMock(return_value=session)),
        patch.object(MessageRepository, "list_by_session", AsyncMock(return_value=[])),
        patch.object(SteeringRepository, "list_by_session", AsyncMock(return_value=rows)),
    ):
        first = SessionStore(database_service=database)
        for index, profile in enumerate(["editor", None, "viewer"]):
            await first.queue_steering(
                "thread",
                AssistantRequest(
                    id=f"steer-{index}",
                    session_id="thread",
                    content=f"Change {index}",
                    message_type="steering",
                    profile=profile,
                ),
            )
        assert [call.kwargs["profile"] for call in create.await_args_list] == [
            "editor",
            None,
            "viewer",
        ]

        restarted = SessionStore(database_service=database)
        assert [
            record["profile"] for record in await restarted.list_pending_steering("thread")
        ] == ["editor", None, "viewer"]
        for profile, expected in [
            ("editor", ["steer-0", "steer-1"]),
            ("viewer", ["steer-1", "steer-2"]),
            ("other", ["steer-1"]),
        ]:
            assert [
                record["id"]
                for record in await restarted.list_pending_steering("thread", profile_name=profile)
            ] == expected
        await restarted.mark_steering_delivered("thread", [])
        assert len(await restarted.list_pending_steering("thread")) == 3
