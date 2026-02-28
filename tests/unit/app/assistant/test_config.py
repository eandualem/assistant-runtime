"""Tests for assistant module configuration."""

import pytest
from pydantic import ValidationError

from lovely_assistant.app.assistant.config import AssistantConfig


class TestAssistantConfigDefaults:
    def test_defaults(self):
        config = AssistantConfig()
        assert config.default_model is None
        assert config.thinking_budget == 10000
        assert config.temperature == 1.0
        assert config.max_turns == 10
        assert config.enable_working_memory is True
        assert config.session_ttl_hours == 24

    def test_custom_values(self):
        config = AssistantConfig(
            default_model="anthropic:claude-sonnet-4-6",
            thinking_budget=5000,
            temperature=0.5,
            max_turns=20,
            enable_working_memory=False,
            session_ttl_hours=48,
        )
        assert config.default_model == "anthropic:claude-sonnet-4-6"
        assert config.thinking_budget == 5000
        assert config.temperature == 0.5
        assert config.max_turns == 20
        assert config.enable_working_memory is False
        assert config.session_ttl_hours == 48


class TestAssistantConfigValidation:
    def test_frozen(self):
        config = AssistantConfig()
        with pytest.raises(ValidationError):
            config.max_turns = 5

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            AssistantConfig(unknown_field="value")

    def test_max_turns_below_min(self):
        with pytest.raises(ValidationError):
            AssistantConfig(max_turns=0)

    def test_max_turns_above_max(self):
        with pytest.raises(ValidationError):
            AssistantConfig(max_turns=51)

    def test_max_turns_boundary_min(self):
        config = AssistantConfig(max_turns=1)
        assert config.max_turns == 1

    def test_max_turns_boundary_max(self):
        config = AssistantConfig(max_turns=50)
        assert config.max_turns == 50

    def test_thinking_budget_below_min(self):
        with pytest.raises(ValidationError):
            AssistantConfig(thinking_budget=0)

    def test_thinking_budget_above_max(self):
        with pytest.raises(ValidationError):
            AssistantConfig(thinking_budget=100001)

    def test_thinking_budget_boundary_min(self):
        config = AssistantConfig(thinking_budget=1)
        assert config.thinking_budget == 1

    def test_thinking_budget_boundary_max(self):
        config = AssistantConfig(thinking_budget=100000)
        assert config.thinking_budget == 100000

    def test_temperature_below_min(self):
        with pytest.raises(ValidationError):
            AssistantConfig(temperature=-0.1)

    def test_temperature_above_max(self):
        with pytest.raises(ValidationError):
            AssistantConfig(temperature=2.1)

    def test_temperature_boundary_min(self):
        config = AssistantConfig(temperature=0.0)
        assert config.temperature == 0.0

    def test_temperature_boundary_max(self):
        config = AssistantConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_session_ttl_hours_below_min(self):
        with pytest.raises(ValidationError):
            AssistantConfig(session_ttl_hours=0)

    def test_session_ttl_hours_above_max(self):
        with pytest.raises(ValidationError):
            AssistantConfig(session_ttl_hours=169)

    def test_session_ttl_hours_boundary_min(self):
        config = AssistantConfig(session_ttl_hours=1)
        assert config.session_ttl_hours == 1

    def test_session_ttl_hours_boundary_max(self):
        config = AssistantConfig(session_ttl_hours=168)
        assert config.session_ttl_hours == 168
