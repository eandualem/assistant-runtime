"""Tests for internal model settings construction."""

import pytest

from lovely_assistant.services.llm._settings import (
    _RESPONSE_MAX_TOKENS,
    build_model_settings,
    normalize_model_id,
    validate_model_id,
)
from lovely_assistant.services.llm.exceptions import ProviderConfigError


class TestNormalizeModelId:
    """normalize_model_id tests — translated from arclio-assistant."""

    def test_empty_string(self):
        assert normalize_model_id("") == ""

    def test_already_canonical(self):
        assert normalize_model_id("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_slash_to_colon(self):
        assert normalize_model_id("anthropic/claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_uppercase_normalized(self):
        assert normalize_model_id("Anthropic:Claude-Sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_whitespace_stripped(self):
        assert (
            normalize_model_id("  anthropic:claude-sonnet-4-6  ") == "anthropic:claude-sonnet-4-6"
        )

    def test_google_prefix_rewritten(self):
        assert normalize_model_id("google:gemini-3-flash") == "google-gla:gemini-3-flash"

    def test_google_slash_rewritten(self):
        assert normalize_model_id("google/gemini-3-flash") == "google-gla:gemini-3-flash"

    def test_google_gla_untouched(self):
        assert normalize_model_id("google-gla:gemini-3-flash") == "google-gla:gemini-3-flash"

    def test_openrouter_untouched(self):
        assert (
            normalize_model_id("openrouter:anthropic/claude-3") == "openrouter:anthropic/claude-3"
        )


class TestValidateModelId:
    """validate_model_id tests."""

    def test_valid_model(self):
        assert validate_model_id("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_normalizes_before_validation(self):
        assert validate_model_id("Anthropic/Claude-Sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_empty_string_raises(self):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            validate_model_id("")

    def test_no_colon_raises(self):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            validate_model_id("just-a-model-name")

    def test_empty_provider_raises(self):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            validate_model_id(":model-name")

    def test_empty_model_raises(self):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            validate_model_id("provider:")


class TestBuildModelSettings:
    """build_model_settings tests — provider-specific settings construction.

    AnthropicModelSettings and OpenRouterModelSettings are TypedDicts,
    so we assert on dict keys rather than isinstance checks.
    """

    def test_anthropic_defaults(self):
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS
        assert settings["anthropic_cache_instructions"] is True
        assert settings["anthropic_cache_tool_definitions"] is True

    def test_anthropic_no_thinking_omits_thinking_param(self):
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6")
        assert isinstance(settings, dict)
        # anthropic_thinking should not be present when thinking is disabled
        assert "anthropic_thinking" not in settings

    def test_anthropic_with_thinking(self):
        settings = build_model_settings(
            model_id="anthropic:claude-sonnet-4-6", thinking_budget=10_000
        )
        assert isinstance(settings, dict)
        assert settings["temperature"] == 1.0  # forced for thinking
        assert settings["max_tokens"] == 10_000 + _RESPONSE_MAX_TOKENS
        assert settings["anthropic_thinking"] == {
            "type": "enabled",
            "budget_tokens": 10_000,
        }

    def test_anthropic_temperature_override(self):
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6", temperature=0.5)
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.5

    def test_anthropic_thinking_ignores_temperature_override(self):
        settings = build_model_settings(
            model_id="anthropic:claude-sonnet-4-6",
            thinking_budget=5000,
            temperature=0.5,
        )
        assert isinstance(settings, dict)
        assert settings["temperature"] == 1.0  # thinking forces 1.0

    def test_anthropic_max_tokens_override(self):
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6", max_tokens=4000)
        assert isinstance(settings, dict)
        assert settings["max_tokens"] == 4000

    def test_anthropic_has_timeout(self):
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6")
        assert isinstance(settings, dict)
        assert "timeout" in settings

    def test_anthropic_has_cache_flags(self):
        """Verify Anthropic settings include cache configuration."""
        settings = build_model_settings(model_id="anthropic:claude-sonnet-4-6")
        assert settings["anthropic_cache_instructions"] is True
        assert settings["anthropic_cache_tool_definitions"] is True

    def test_openrouter_defaults(self):
        settings = build_model_settings(model_id="openrouter:anthropic/claude-3")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS
        assert settings["openrouter_provider"] == {
            "data_collection": "deny",
            "require_parameters": True,
        }

    def test_openrouter_with_thinking(self):
        settings = build_model_settings(
            model_id="openrouter:anthropic/claude-3", thinking_budget=10_000
        )
        assert isinstance(settings, dict)
        assert settings["openrouter_reasoning"] == {"effort": "high"}
        # OpenRouter doesn't force temperature 1.0 for thinking
        assert settings["temperature"] == 0.1

    def test_generic_provider_returns_dict(self):
        settings = build_model_settings(model_id="openai:gpt-4o")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS
        # Generic dict should NOT have provider-specific keys
        assert "anthropic_cache_instructions" not in settings
        assert "openrouter_provider" not in settings

    def test_generic_provider_thinking_uses_base_max_tokens(self):
        """Generic providers ignore thinking_budget in max_tokens calculation."""
        settings = build_model_settings(model_id="openai:gpt-4o", thinking_budget=10_000)
        assert isinstance(settings, dict)
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS  # not inflated
