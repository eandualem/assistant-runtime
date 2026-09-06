from __future__ import annotations

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from assistant_runtime.services.database.models import MessageORM, SessionORM, SteeringORM


class TestSessionORM:
    def test_session_columns_drop_flat_message_history(self) -> None:
        columns = {column.name for column in SessionORM.__table__.columns}

        assert columns == {
            "id",
            "title",
            "turn_number",
            "working_memory",
            "pending_action",
            "telegram_chat_id",
            "owner_id",
            "telegram_bound_at",
            "created_at",
            "updated_at",
            "expires_at",
        }
        assert "message_history" not in columns

    def test_session_column_types(self) -> None:
        assert isinstance(SessionORM.__table__.columns["id"].type, String)
        assert isinstance(SessionORM.__table__.columns["title"].type, Text)
        assert isinstance(SessionORM.__table__.columns["turn_number"].type, Integer)
        assert isinstance(SessionORM.__table__.columns["working_memory"].type, JSONB)
        assert isinstance(SessionORM.__table__.columns["created_at"].type, DateTime)


class TestMessageORM:
    def test_message_columns_match_tree_schema(self) -> None:
        columns = MessageORM.__table__.columns

        assert {column.name for column in columns} == {
            "id",
            "session_id",
            "parent_id",
            "role",
            "message_type",
            "content",
            "segments",
            "usage",
            "created_at",
        }
        assert isinstance(columns["segments"].type, JSONB)
        assert isinstance(columns["usage"].type, JSONB)
        assert isinstance(columns["content"].type, Text)

    def test_message_constraints_and_indexes_exist(self) -> None:
        constraint_names = {constraint.name for constraint in MessageORM.__table__.constraints}
        index_names = {index.name for index in MessageORM.__table__.indexes}

        assert any("ck_messages_role_valid" in name for name in constraint_names)
        assert any("ck_messages_message_type_valid" in name for name in constraint_names)
        assert "ix_messages_parent_id" in index_names
        assert "ix_messages_session_id_created_at" in index_names
        assert "uq_messages_single_root_per_session" in index_names


class TestSteeringORM:
    def test_steering_columns_match_out_of_band_schema(self) -> None:
        columns = SteeringORM.__table__.columns

        assert {column.name for column in columns} == {
            "id",
            "session_id",
            "content",
            "status",
            "created_at",
            "delivered_at",
        }
        assert isinstance(columns["content"].type, Text)
        assert isinstance(columns["status"].type, String)

    def test_steering_indexes_exist(self) -> None:
        constraint_names = {constraint.name for constraint in SteeringORM.__table__.constraints}
        index_names = {index.name for index in SteeringORM.__table__.indexes}

        assert any("ck_steering_status_valid" in name for name in constraint_names)
        assert "ix_steering_session_id_created_at" in index_names
        assert "ix_steering_session_id_status_created_at" in index_names
