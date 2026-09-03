"""Tests for RuntimeSettings, EffectiveConfig, and resolve_effective_config."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.models import RequestConfigOverride
from assistant_runtime.app.settings import (
    EffectiveConfig,
    RuntimeSettings,
    resolve_effective_config,
)


class TestRuntimeSettings:
    def test_initial_state_no_overrides(self):
        frozen = AssistantConfig()
        rs = RuntimeSettings(frozen_config=frozen)
        resp = rs.to_response_dict()
        assert resp["updated_at"] is None
        for field_info in resp["values"].values():
            assert field_info["source"] == "config_default"

    @pytest.mark.asyncio
    async def test_update_single_field(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(temperature=0.5)
        resp = rs.to_response_dict()
        assert resp["values"]["temperature"]["value"] == 0.5
        assert resp["values"]["temperature"]["source"] == "runtime"
        assert resp["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(temperature=0.7, max_turns=20, default_model="anthropic:claude-haiku-4-5")
        resp = rs.to_response_dict()
        assert resp["values"]["temperature"]["value"] == 0.7
        assert resp["values"]["max_turns"]["value"] == 20
        assert resp["values"]["default_model"]["value"] == "anthropic:claude-haiku-4-5"
        assert resp["values"]["temperature"]["source"] == "runtime"
        assert resp["values"]["max_turns"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_clear_with_none(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(temperature=0.5)
        assert rs.to_response_dict()["values"]["temperature"]["source"] == "runtime"
        await rs.update(temperature=None)
        assert rs.to_response_dict()["values"]["temperature"]["source"] == "config_default"

    @pytest.mark.asyncio
    async def test_reject_unknown_fields(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="Unknown settings field"):
            await rs.update(nonexistent_field=42)

    @pytest.mark.asyncio
    async def test_reject_temperature_out_of_range(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="temperature must be between"):
            await rs.update(temperature=3.0)

    @pytest.mark.asyncio
    async def test_reject_temperature_negative(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="temperature must be between"):
            await rs.update(temperature=-0.1)

    @pytest.mark.asyncio
    async def test_reject_thinking_budget_out_of_range(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="thinking_budget must be between"):
            await rs.update(thinking_budget=200_000)

    @pytest.mark.asyncio
    async def test_reject_thinking_budget_zero(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="thinking_budget must be between"):
            await rs.update(thinking_budget=0)

    @pytest.mark.asyncio
    async def test_reject_max_turns_out_of_range(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="max_turns must be between"):
            await rs.update(max_turns=100)

    def test_to_response_dict_defaults(self):
        frozen = AssistantConfig(default_model=None, thinking_budget=None, max_turns=10)
        rs = RuntimeSettings(frozen_config=frozen)
        resp = rs.to_response_dict()
        assert resp["values"]["default_model"]["value"] is None
        assert resp["values"]["max_turns"]["value"] == 10
        assert resp["values"]["enable_working_memory"]["value"] is True

    @pytest.mark.asyncio
    async def test_updated_at_tracking(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        assert rs.to_response_dict()["updated_at"] is None
        await rs.update(temperature=0.5)
        first_update = rs.to_response_dict()["updated_at"]
        assert first_update is not None
        await rs.update(max_turns=5)
        second_update = rs.to_response_dict()["updated_at"]
        assert second_update is not None
        assert second_update >= first_update

    def test_valid_fields_contains_all_11_fields(self):
        expected = {
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
        }
        assert expected == RuntimeSettings._VALID_FIELDS
        assert len(RuntimeSettings._VALID_FIELDS) == 11

    @pytest.mark.asyncio
    async def test_update_summarization_model(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(summarization_model="openai:gpt-4o-mini")
        resp = rs.to_response_dict()
        assert resp["values"]["summarization_model"]["value"] == "openai:gpt-4o-mini"
        assert resp["values"]["summarization_model"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_update_subagent_thinking_budget(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(subagent_thinking_budget=5000)
        resp = rs.to_response_dict()
        assert resp["values"]["subagent_thinking_budget"]["value"] == 5000
        assert resp["values"]["subagent_thinking_budget"]["source"] == "runtime"

    @pytest.mark.asyncio
    async def test_reject_subagent_thinking_budget_zero(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="subagent_thinking_budget must be between"):
            await rs.update(subagent_thinking_budget=0)

    @pytest.mark.asyncio
    async def test_reject_subagent_thinking_budget_too_large(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        with pytest.raises(ValueError, match="subagent_thinking_budget must be between"):
            await rs.update(subagent_thinking_budget=200_000)

    @pytest.mark.asyncio
    async def test_get_returns_runtime_value_when_overridden(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.update(temperature=0.5)
        assert rs.get("temperature", 1.0) == 0.5

    def test_get_returns_frozen_default_when_not_overridden(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        assert rs.get("temperature", 1.0) == 1.0

    def test_get_returns_none_frozen_default(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        assert rs.get("summarization_model", None) is None

    def test_to_response_dict_includes_all_11_fields(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        resp = rs.to_response_dict()
        expected_fields = {
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
        }
        assert set(resp["values"].keys()) == expected_fields


class TestRuntimeSettingsDB:
    """Tests for DB persistence in RuntimeSettings."""

    def _make_mock_db(self, *, healthy: bool = True) -> MagicMock:
        """Create a mock DatabaseService with session_context as async context manager."""
        mock_db = MagicMock()
        mock_db._healthy = healthy
        mock_session = AsyncMock()

        @asynccontextmanager
        async def fake_session_context():
            yield mock_session

        mock_db.session_context = fake_session_context
        mock_db._mock_session = mock_session  # stash for assertions
        return mock_db

    # -- __init__ tests --

    def test_init_accepts_database_service(self):
        frozen = AssistantConfig()
        mock_db = MagicMock()
        rs = RuntimeSettings(frozen_config=frozen, database_service=mock_db)
        assert rs._db is mock_db

    def test_init_database_service_defaults_to_none(self):
        frozen = AssistantConfig()
        rs = RuntimeSettings(frozen_config=frozen)
        assert rs._db is None

    # -- load_from_db tests --

    @pytest.mark.asyncio
    async def test_load_from_db_skips_when_no_db(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        await rs.load_from_db()  # Should not raise
        assert len(rs._overridden) == 0

    @pytest.mark.asyncio
    async def test_load_from_db_loads_non_null_fields(self):
        mock_row = MagicMock()
        mock_row.default_model = "anthropic:claude-sonnet-4-6"
        mock_row.thinking_budget = None
        mock_row.temperature = 0.5
        mock_row.max_turns = None
        mock_row.enable_working_memory = None
        mock_row.summarization_model = None
        mock_row.working_memory_model = None
        mock_row.default_image_model = None
        mock_row.default_video_model = None
        mock_row.subagent_model = None
        mock_row.subagent_thinking_budget = None
        mock_row.updated_at = datetime(2026, 2, 20, tzinfo=UTC)

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_row)
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.load_from_db()

        assert "default_model" in rs._overridden
        assert "temperature" in rs._overridden
        assert rs._default_model == "anthropic:claude-sonnet-4-6"
        assert rs._temperature == 0.5
        assert "thinking_budget" not in rs._overridden
        assert "max_turns" not in rs._overridden
        assert "enable_working_memory" not in rs._overridden
        assert rs._updated_at == datetime(2026, 2, 20, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_load_from_db_no_row_returns_no_overrides(self):
        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=None)
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.load_from_db()

        assert len(rs._overridden) == 0
        assert rs._updated_at is None

    @pytest.mark.asyncio
    async def test_load_from_db_handles_exception_gracefully(self):
        mock_db = MagicMock()
        mock_db._healthy = True

        @asynccontextmanager
        async def failing_session_context():
            raise RuntimeError("DB connection lost")
            yield  # noqa: F541 — required for generator syntax

        mock_db.session_context = failing_session_context

        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)
        await rs.load_from_db()  # Should not raise
        assert len(rs._overridden) == 0

    # -- _persist_to_db tests --

    @pytest.mark.asyncio
    async def test_persist_to_db_skips_when_no_db(self):
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        # update() calls _persist_to_db internally — should succeed silently
        await rs.update(temperature=0.5)
        resp = rs.to_response_dict()
        assert resp["values"]["temperature"]["value"] == 0.5

    @pytest.mark.asyncio
    async def test_update_calls_persist(self):
        mock_repo = MagicMock()
        mock_repo.save = AsyncMock()
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            result = await rs.update(temperature=0.5)

        # session_context was used — the repo was constructed and save was called
        mock_settings_repo_cls.assert_called_once()
        mock_repo.save.assert_awaited_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_persist_sends_full_state(self):
        mock_repo = MagicMock()
        mock_repo.save = AsyncMock()
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.update(temperature=0.8, max_turns=20)

        # Verify save() was called with a dict containing all fields
        save_call_args = mock_repo.save.call_args
        state_dict = save_call_args[0][0]

        # Overridden fields have their values
        assert state_dict["temperature"] == 0.8
        assert state_dict["max_turns"] == 20
        # Non-overridden fields are None (clears old DB values)
        assert state_dict["default_model"] is None
        assert state_dict["thinking_budget"] is None
        assert state_dict["enable_working_memory"] is None
        # All valid fields are present
        assert set(state_dict.keys()) == RuntimeSettings._VALID_FIELDS

    @pytest.mark.asyncio
    async def test_load_from_db_recovers_after_degraded_startup(self):
        """DB was unhealthy at start but session_context works -- settings load."""
        mock_row = MagicMock()
        mock_row.default_model = "anthropic:claude-sonnet-4-6"
        mock_row.thinking_budget = None
        mock_row.temperature = None
        mock_row.max_turns = None
        mock_row.enable_working_memory = None
        mock_row.summarization_model = None
        mock_row.working_memory_model = None
        mock_row.default_image_model = None
        mock_row.default_video_model = None
        mock_row.subagent_model = None
        mock_row.subagent_thinking_budget = None
        mock_row.updated_at = None

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_row)
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=False)  # unhealthy at start
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.load_from_db()

        # Still loaded because we no longer check _healthy
        assert "default_model" in rs._overridden
        assert rs._default_model == "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_persist_returns_false_on_failure(self):
        """DB that throws returns False from update()."""
        mock_db = MagicMock()

        @asynccontextmanager
        async def failing_session_context():
            raise RuntimeError("DB connection lost")
            yield  # noqa: F541

        mock_db.session_context = failing_session_context
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)
        result = await rs.update(temperature=0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_persist_returns_true_on_success(self):
        """DB that works returns True from update()."""
        mock_repo = MagicMock()
        mock_repo.save = AsyncMock()
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            result = await rs.update(temperature=0.5)

        assert result is True

    @pytest.mark.asyncio
    async def test_persist_returns_false_when_no_db(self):
        """No DB configured returns False from update()."""
        rs = RuntimeSettings(frozen_config=AssistantConfig())
        result = await rs.update(temperature=0.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_load_from_db_loads_new_fields(self):
        mock_row = MagicMock()
        mock_row.default_model = None
        mock_row.thinking_budget = None
        mock_row.temperature = None
        mock_row.max_turns = None
        mock_row.enable_working_memory = None
        mock_row.summarization_model = "openai:gpt-4o-mini"
        mock_row.working_memory_model = None
        mock_row.default_image_model = None
        mock_row.default_video_model = None
        mock_row.subagent_model = None
        mock_row.subagent_thinking_budget = 5000
        mock_row.updated_at = datetime(2026, 2, 21, tzinfo=UTC)

        mock_repo = MagicMock()
        mock_repo.get = AsyncMock(return_value=mock_row)
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.load_from_db()

        assert "summarization_model" in rs._overridden
        assert rs._summarization_model == "openai:gpt-4o-mini"
        assert "subagent_thinking_budget" in rs._overridden
        assert rs._subagent_thinking_budget == 5000
        assert "working_memory_model" not in rs._overridden
        assert "default_image_model" not in rs._overridden
        assert "default_video_model" not in rs._overridden
        assert "subagent_model" not in rs._overridden
        assert rs._updated_at == datetime(2026, 2, 21, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_persist_sends_all_11_fields(self):
        mock_repo = MagicMock()
        mock_repo.save = AsyncMock()
        mock_settings_repo_cls = MagicMock(return_value=mock_repo)

        mock_db = self._make_mock_db(healthy=True)
        rs = RuntimeSettings(frozen_config=AssistantConfig(), database_service=mock_db)

        with patch(
            "assistant_runtime.services.database.repositories.SettingsRepository",
            mock_settings_repo_cls,
        ):
            await rs.update(summarization_model="openai:gpt-4o-mini")

        save_call_args = mock_repo.save.call_args
        state_dict = save_call_args[0][0]

        # Overridden field has its value
        assert state_dict["summarization_model"] == "openai:gpt-4o-mini"
        # Non-overridden fields are None
        assert state_dict["default_model"] is None
        assert state_dict["subagent_thinking_budget"] is None
        # All 11 valid fields are present
        assert set(state_dict.keys()) == RuntimeSettings._VALID_FIELDS
        assert len(state_dict) == 11


class TestResolveEffectiveConfig:
    def test_all_defaults(self):
        frozen = AssistantConfig()
        effective = resolve_effective_config(frozen)
        assert effective.default_model is None
        assert effective.thinking_budget == 10000
        assert effective.temperature == 1.0
        assert effective.max_turns == 10
        assert effective.enable_working_memory is True

    @pytest.mark.asyncio
    async def test_runtime_overrides_default(self):
        frozen = AssistantConfig()
        rs = RuntimeSettings(frozen_config=frozen)
        await rs.update(temperature=0.8, max_turns=20)
        effective = resolve_effective_config(frozen, rs)
        assert effective.temperature == 0.8
        assert effective.max_turns == 20

    @pytest.mark.asyncio
    async def test_per_request_overrides_runtime(self):
        frozen = AssistantConfig()
        rs = RuntimeSettings(frozen_config=frozen)
        await rs.update(temperature=0.8)
        per_req = RequestConfigOverride(temperature=1.5)
        effective = resolve_effective_config(frozen, rs, per_req)
        assert effective.temperature == 1.5

    def test_none_falls_through(self):
        frozen = AssistantConfig(default_model="anthropic:claude-sonnet-4-6")
        per_req = RequestConfigOverride(default_model=None)
        effective = resolve_effective_config(frozen, None, per_req)
        # None in per_request doesn't override — falls through to frozen
        assert effective.default_model == "anthropic:claude-sonnet-4-6"

    def test_none_runtime_settings(self):
        frozen = AssistantConfig(max_turns=15)
        effective = resolve_effective_config(frozen, None, None)
        assert effective.max_turns == 15

    def test_none_per_request(self):
        frozen = AssistantConfig()
        effective = resolve_effective_config(frozen, None, None)
        assert effective.max_turns == 10

    def test_effective_config_is_frozen(self):
        effective = EffectiveConfig(
            default_model=None,
            thinking_budget=None,
            temperature=None,
            max_turns=10,
            enable_working_memory=True,
            summarization_model=None,
            working_memory_model=None,
            default_image_model=None,
            default_video_model=None,
            subagent_model=None,
            subagent_thinking_budget=None,
        )
        with pytest.raises(AttributeError):
            effective.temperature = 0.5  # type: ignore[misc]
