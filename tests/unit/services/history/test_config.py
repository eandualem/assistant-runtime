"""Tests for HistoryConfig validation, defaults, and bounds."""

import pytest
from pydantic import ValidationError

from assistant_runtime.services.history.config import HistoryConfig


class TestHistoryConfigDefaults:
    def test_default_values(self):
        config = HistoryConfig()
        assert config.token_budget == 100_000
        assert config.retain_recent == 5
        assert config.protect_recent_tool_results == 3
        assert config.summarization_model is None
        assert config.working_memory_enabled is True
        assert config.working_memory_model is None
        assert config.message_truncation_limit == 1000
        assert config.max_memory_entries == 15

    def test_custom_values(self):
        config = HistoryConfig(
            token_budget=50_000,
            retain_recent=10,
            protect_recent_tool_results=5,
            summarization_model="openai:gpt-4o-mini",
            working_memory_enabled=False,
            working_memory_model="openai:gpt-4o",
            message_truncation_limit=2000,
            max_memory_entries=20,
        )
        assert config.token_budget == 50_000
        assert config.retain_recent == 10
        assert config.protect_recent_tool_results == 5
        assert config.summarization_model == "openai:gpt-4o-mini"
        assert config.working_memory_enabled is False
        assert config.working_memory_model == "openai:gpt-4o"
        assert config.message_truncation_limit == 2000
        assert config.max_memory_entries == 20


class TestHistoryConfigValidation:
    def test_frozen(self):
        config = HistoryConfig()
        with pytest.raises(ValidationError):
            config.token_budget = 999

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            HistoryConfig(nonexistent_field="value")

    def test_token_budget_min(self):
        with pytest.raises(ValidationError):
            HistoryConfig(token_budget=4999)

    def test_token_budget_max(self):
        with pytest.raises(ValidationError):
            HistoryConfig(token_budget=500_001)

    def test_token_budget_boundary_min(self):
        config = HistoryConfig(token_budget=5000)
        assert config.token_budget == 5000

    def test_token_budget_boundary_max(self):
        config = HistoryConfig(token_budget=500_000)
        assert config.token_budget == 500_000

    def test_retain_recent_min(self):
        with pytest.raises(ValidationError):
            HistoryConfig(retain_recent=0)

    def test_retain_recent_max(self):
        with pytest.raises(ValidationError):
            HistoryConfig(retain_recent=51)

    def test_protect_recent_tool_results_min(self):
        config = HistoryConfig(protect_recent_tool_results=0)
        assert config.protect_recent_tool_results == 0

    def test_protect_recent_tool_results_max(self):
        with pytest.raises(ValidationError):
            HistoryConfig(protect_recent_tool_results=21)

    def test_message_truncation_limit_min(self):
        with pytest.raises(ValidationError):
            HistoryConfig(message_truncation_limit=99)

    def test_message_truncation_limit_max(self):
        with pytest.raises(ValidationError):
            HistoryConfig(message_truncation_limit=10001)

    def test_max_memory_entries_min(self):
        with pytest.raises(ValidationError):
            HistoryConfig(max_memory_entries=4)

    def test_max_memory_entries_max(self):
        with pytest.raises(ValidationError):
            HistoryConfig(max_memory_entries=51)
