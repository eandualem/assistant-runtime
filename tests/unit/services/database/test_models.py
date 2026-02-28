"""Tests for ORM models — column types, defaults, and constraints."""

from __future__ import annotations

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from lovely_assistant.services.database.base import Base
from lovely_assistant.services.database.models import ArtifactORM, SessionORM, UserSettingsORM


class TestSessionORMTableName:
    def test_table_name_is_sessions(self):
        assert SessionORM.__tablename__ == "sessions"


class TestSessionORMInheritance:
    def test_is_subclass_of_base(self):
        assert issubclass(SessionORM, Base)


class TestSessionORMColumns:
    """Verify all expected columns exist with correct types and constraints."""

    def _col(self, name: str):
        """Helper to retrieve a column from the model's table."""
        return SessionORM.__table__.columns[name]

    def test_has_all_expected_columns(self):
        expected = {
            "id",
            "title",
            "turn_number",
            "message_history",
            "working_memory",
            "created_at",
            "updated_at",
            "expires_at",
        }
        actual = {c.name for c in SessionORM.__table__.columns}
        assert actual == expected

    # --- id ---

    def test_id_is_string_64(self):
        col = self._col("id")
        assert isinstance(col.type, String)
        assert col.type.length == 64

    def test_id_is_primary_key(self):
        col = self._col("id")
        assert col.primary_key is True

    # --- title ---

    def test_title_is_text(self):
        col = self._col("title")
        assert isinstance(col.type, Text)

    def test_title_is_nullable(self):
        col = self._col("title")
        assert col.nullable is True

    # --- turn_number ---

    def test_turn_number_is_integer(self):
        col = self._col("turn_number")
        assert isinstance(col.type, Integer)

    def test_turn_number_has_default(self):
        col = self._col("turn_number")
        assert col.default is not None
        assert col.default.arg == 0

    # --- message_history ---

    def test_message_history_is_jsonb(self):
        col = self._col("message_history")
        assert isinstance(col.type, JSONB)

    # --- working_memory ---

    def test_working_memory_is_jsonb(self):
        col = self._col("working_memory")
        assert isinstance(col.type, JSONB)

    def test_working_memory_is_nullable(self):
        col = self._col("working_memory")
        assert col.nullable is True

    # --- created_at ---

    def test_created_at_is_datetime_with_timezone(self):
        col = self._col("created_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_created_at_has_server_default(self):
        col = self._col("created_at")
        assert col.server_default is not None

    # --- updated_at ---

    def test_updated_at_is_datetime_with_timezone(self):
        col = self._col("updated_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_updated_at_has_server_default(self):
        col = self._col("updated_at")
        assert col.server_default is not None

    def test_updated_at_has_onupdate(self):
        col = self._col("updated_at")
        assert col.onupdate is not None

    # --- expires_at ---

    def test_expires_at_is_datetime_with_timezone(self):
        col = self._col("expires_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_expires_at_has_server_default(self):
        col = self._col("expires_at")
        assert col.server_default is not None

    def test_expires_at_is_not_nullable(self):
        col = self._col("expires_at")
        assert col.nullable is False


class TestUserSettingsORMTableName:
    def test_table_name_is_user_settings(self):
        assert UserSettingsORM.__tablename__ == "user_settings"


class TestUserSettingsORMInheritance:
    def test_is_subclass_of_base(self):
        assert issubclass(UserSettingsORM, Base)


class TestUserSettingsORMColumns:
    """Verify all expected columns exist with correct types and constraints."""

    def _col(self, name: str):
        """Helper to retrieve a column from the model's table."""
        return UserSettingsORM.__table__.columns[name]

    def test_has_all_expected_columns(self):
        expected = {
            "id",
            "default_model",
            "thinking_budget",
            "temperature",
            "max_turns",
            "enable_working_memory",
            "summarization_model",
            "working_memory_model",
            "default_image_model",
            "default_video_model",
            "subagent_model",
            "subagent_thinking_budget",
            "updated_at",
        }
        actual = {c.name for c in UserSettingsORM.__table__.columns}
        assert actual == expected

    # --- id ---

    def test_id_is_string_36(self):
        col = self._col("id")
        assert isinstance(col.type, String)
        assert col.type.length == 36

    def test_id_is_primary_key(self):
        col = self._col("id")
        assert col.primary_key is True

    # --- default_model ---

    def test_default_model_is_text(self):
        col = self._col("default_model")
        assert isinstance(col.type, Text)

    def test_default_model_is_nullable(self):
        col = self._col("default_model")
        assert col.nullable is True

    # --- thinking_budget ---

    def test_thinking_budget_is_integer(self):
        col = self._col("thinking_budget")
        assert isinstance(col.type, Integer)

    def test_thinking_budget_is_nullable(self):
        col = self._col("thinking_budget")
        assert col.nullable is True

    # --- temperature ---

    def test_temperature_is_float(self):
        col = self._col("temperature")
        assert isinstance(col.type, Float)

    def test_temperature_is_nullable(self):
        col = self._col("temperature")
        assert col.nullable is True

    # --- max_turns ---

    def test_max_turns_is_integer(self):
        col = self._col("max_turns")
        assert isinstance(col.type, Integer)

    def test_max_turns_is_nullable(self):
        col = self._col("max_turns")
        assert col.nullable is True

    # --- enable_working_memory ---

    def test_enable_working_memory_is_boolean(self):
        col = self._col("enable_working_memory")
        assert isinstance(col.type, Boolean)

    def test_enable_working_memory_is_nullable(self):
        col = self._col("enable_working_memory")
        assert col.nullable is True

    # --- updated_at ---

    def test_updated_at_is_datetime_with_timezone(self):
        col = self._col("updated_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_updated_at_has_server_default(self):
        col = self._col("updated_at")
        assert col.server_default is not None

    def test_updated_at_has_onupdate(self):
        col = self._col("updated_at")
        assert col.onupdate is not None


class TestArtifactORMTableName:
    def test_table_name_is_artifacts(self):
        assert ArtifactORM.__tablename__ == "artifacts"


class TestArtifactORMInheritance:
    def test_is_subclass_of_base(self):
        assert issubclass(ArtifactORM, Base)


class TestArtifactORMColumns:
    """Verify all expected columns exist with correct types and constraints."""

    def _col(self, name: str):
        """Helper to retrieve a column from the model's table."""
        return ArtifactORM.__table__.columns[name]

    def test_has_all_expected_columns(self):
        expected = {
            "id",
            "name",
            "content",
            "version",
            "is_active",
            "proposed_by",
            "created_at",
        }
        actual = {c.name for c in ArtifactORM.__table__.columns}
        assert actual == expected

    # --- id ---

    def test_id_is_integer(self):
        col = self._col("id")
        assert isinstance(col.type, Integer)

    def test_id_is_primary_key(self):
        col = self._col("id")
        assert col.primary_key is True

    # --- name ---

    def test_name_is_string_64(self):
        col = self._col("name")
        assert isinstance(col.type, String)
        assert col.type.length == 64

    def test_name_is_not_nullable(self):
        col = self._col("name")
        assert col.nullable is False

    # --- content ---

    def test_content_is_text(self):
        col = self._col("content")
        assert isinstance(col.type, Text)

    def test_content_is_not_nullable(self):
        col = self._col("content")
        assert col.nullable is False

    # --- version ---

    def test_version_is_integer(self):
        col = self._col("version")
        assert isinstance(col.type, Integer)

    def test_version_is_not_nullable(self):
        col = self._col("version")
        assert col.nullable is False

    def test_version_has_server_default(self):
        col = self._col("version")
        assert col.server_default is not None

    # --- is_active ---

    def test_is_active_is_boolean(self):
        col = self._col("is_active")
        assert isinstance(col.type, Boolean)

    def test_is_active_is_not_nullable(self):
        col = self._col("is_active")
        assert col.nullable is False

    def test_is_active_has_server_default(self):
        col = self._col("is_active")
        assert col.server_default is not None

    # --- proposed_by ---

    def test_proposed_by_is_string_32(self):
        col = self._col("proposed_by")
        assert isinstance(col.type, String)
        assert col.type.length == 32

    def test_proposed_by_is_not_nullable(self):
        col = self._col("proposed_by")
        assert col.nullable is False

    def test_proposed_by_has_server_default(self):
        col = self._col("proposed_by")
        assert col.server_default is not None

    # --- created_at ---

    def test_created_at_is_datetime_with_timezone(self):
        col = self._col("created_at")
        assert isinstance(col.type, DateTime)
        assert col.type.timezone is True

    def test_created_at_has_server_default(self):
        col = self._col("created_at")
        assert col.server_default is not None


class TestArtifactORMConstraints:
    """Verify constraints on the artifacts table."""

    def test_has_unique_name_version_constraint(self):
        constraints = ArtifactORM.__table__.constraints
        unique_constraints = [
            c for c in constraints if hasattr(c, "columns") and len(c.columns) == 2
        ]
        name_version_uq = [
            c for c in unique_constraints if {col.name for col in c.columns} == {"name", "version"}
        ]
        assert len(name_version_uq) == 1

    def test_has_name_is_active_index(self):
        indexes = ArtifactORM.__table__.indexes
        matching = [
            idx for idx in indexes if {col.name for col in idx.columns} == {"name", "is_active"}
        ]
        assert len(matching) == 1
