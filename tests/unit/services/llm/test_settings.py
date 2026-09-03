"""Tests for internal model settings construction."""

import pytest

from assistant_runtime.services.llm._settings import (
    _RESPONSE_MAX_TOKENS,
    build_model_settings,
    validate_model_id,
)
from assistant_runtime.services.llm.exceptions import ProviderConfigError


class TestValidateModelId:
    """validate_model_id tests — strict validation, no normalization."""

    def test_valid_model(self):
        assert validate_model_id("anthropic:claude-sonnet-4-6") == "anthropic:claude-sonnet-4-6"

    def test_openrouter_with_slash_after_colon(self):
        assert validate_model_id("openrouter:anthropic/claude-3") == "openrouter:anthropic/claude-3"

    def test_google_prefix_accepted(self):
        assert validate_model_id("google:gemini-3-flash") == "google:gemini-3-flash"

    def test_whitespace_stripped(self):
        assert validate_model_id("  anthropic:claude-sonnet-4-6  ") == "anthropic:claude-sonnet-4-6"

    def test_empty_string_raises(self):
        with pytest.raises(ProviderConfigError, match="cannot be empty"):
            validate_model_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ProviderConfigError, match="cannot be empty"):
            validate_model_id("   ")

    def test_uppercase_raises(self):
        with pytest.raises(ProviderConfigError, match="must be lowercase"):
            validate_model_id("Anthropic:claude-sonnet-4-6")

    def test_slash_separator_raises(self):
        with pytest.raises(ProviderConfigError, match="colon separator"):
            validate_model_id("anthropic/claude-sonnet-4-6")

    def test_legacy_google_gla_prefix_raises(self):
        with pytest.raises(ProviderConfigError, match="google:"):
            validate_model_id("google-gla:gemini-3-flash")

    def test_legacy_google_vertex_prefix_raises(self):
        with pytest.raises(ProviderConfigError, match="google-cloud:"):
            validate_model_id("google-vertex:gemini-3-flash")

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

    def test_anthropic_with_thinking_adaptive_model(self):
        """Sonnet 4.6+ / Opus 4.6+ use adaptive thinking; the budget maps to an effort level."""
        settings = build_model_settings(
            model_id="anthropic:claude-sonnet-4-6", thinking_budget=10_000
        )
        assert isinstance(settings, dict)
        assert settings["temperature"] == 1.0  # forced for thinking on models that allow sampling
        assert settings["max_tokens"] == 10_000 + _RESPONSE_MAX_TOKENS
        assert settings["anthropic_thinking"] == {"type": "adaptive"}
        assert settings["anthropic_effort"] == "medium"

    def test_anthropic_with_thinking_budget_model(self):
        """Pre-4.6 models keep the explicit budget_tokens shape."""
        settings = build_model_settings(
            model_id="anthropic:claude-haiku-4-5", thinking_budget=10_000
        )
        assert settings["anthropic_thinking"] == {
            "type": "enabled",
            "budget_tokens": 10_000,
        }
        assert "anthropic_effort" not in settings

    def test_anthropic_effort_tiers(self):
        for budget, effort in (
            (2_000, "low"),
            (10_000, "medium"),
            (30_000, "high"),
            (60_000, "xhigh"),
        ):
            settings = build_model_settings(
                model_id="anthropic:claude-opus-5", thinking_budget=budget
            )
            assert settings["anthropic_effort"] == effort, budget
        # Sonnet 4.6 has no xhigh tier: large budgets cap at high
        settings = build_model_settings(
            model_id="anthropic:claude-sonnet-4-6", thinking_budget=60_000
        )
        assert settings["anthropic_effort"] == "high"

    def test_anthropic_no_sampling_models_omit_temperature(self):
        """Opus 4.7+, Sonnet 5 and Fable reject temperature; it must not be sent."""
        for model in (
            "anthropic:claude-opus-5",
            "anthropic:claude-sonnet-5",
            "anthropic:claude-fable-5-1",
        ):
            settings = build_model_settings(model_id=model, temperature=0.5)
            assert "temperature" not in settings, model
            with_thinking = build_model_settings(model_id=model, thinking_budget=5_000)
            assert "temperature" not in with_thinking, model
            assert with_thinking["anthropic_thinking"] == {"type": "adaptive"}

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

    def test_openai_gpt5_defaults_without_reasoning(self):
        settings = build_model_settings(model_id="openai:gpt-5.4")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS
        assert settings["openai_previous_response_id"] == "auto"
        assert settings["openai_send_reasoning_ids"] is False
        assert "openai_reasoning_effort" not in settings
        assert "openai_reasoning_summary" not in settings

    @pytest.mark.parametrize(
        ("thinking_budget", "expected_effort"),
        [
            (1_000, "low"),
            (5_000, "medium"),
            (20_000, "high"),
            (40_000, "xhigh"),
        ],
    )
    def test_openai_gpt5_thinking_budget_maps_reasoning_effort(
        self, thinking_budget, expected_effort
    ):
        settings = build_model_settings(
            model_id="openai:gpt-5.4",
            thinking_budget=thinking_budget,
        )
        assert isinstance(settings, dict)
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS
        assert settings["openai_previous_response_id"] == "auto"
        assert settings["openai_send_reasoning_ids"] is False
        assert settings["openai_reasoning_effort"] == expected_effort
        assert settings["openai_reasoning_summary"] == "detailed"

    def test_openai_gpt5_pro_uses_fixed_high_reasoning_effort(self):
        settings = build_model_settings(
            model_id="openai:gpt-5.4-pro",
            thinking_budget=1_000,
        )
        assert isinstance(settings, dict)
        assert settings["openai_previous_response_id"] == "auto"
        assert settings["openai_send_reasoning_ids"] is False
        assert settings["openai_reasoning_effort"] == "high"
        assert settings["openai_reasoning_summary"] == "detailed"


class TestBuildModelSettingsGoogle:
    """Google provider settings — GoogleModelSettings with thinking config."""

    def test_google_gla_defaults(self):
        settings = build_model_settings(model_id="google:gemini-3-flash-preview")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS

    def test_google_gla_no_thinking_omits_config(self):
        settings = build_model_settings(model_id="google:gemini-3-flash-preview")
        assert isinstance(settings, dict)
        assert "google_thinking_config" not in settings

    def test_google_gla_with_thinking(self):
        settings = build_model_settings(
            model_id="google:gemini-3-flash-preview", thinking_budget=10_000
        )
        assert isinstance(settings, dict)
        assert settings["google_thinking_config"] == {
            "include_thoughts": True,
            "thinking_budget": 10_000,
        }

    def test_google_thinking_does_not_inflate_max_tokens(self):
        """Google thinking budget is separate — max_tokens stays at base."""
        settings = build_model_settings(
            model_id="google:gemini-3-flash-preview", thinking_budget=10_000
        )
        assert isinstance(settings, dict)
        assert settings["max_tokens"] == _RESPONSE_MAX_TOKENS

    def test_google_temperature_override(self):
        settings = build_model_settings(model_id="google:gemini-3-flash-preview", temperature=0.7)
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.7

    def test_google_max_tokens_override(self):
        settings = build_model_settings(model_id="google:gemini-3-flash-preview", max_tokens=4000)
        assert isinstance(settings, dict)
        assert settings["max_tokens"] == 4000

    def test_google_no_anthropic_or_openrouter_keys(self):
        """Google settings should not have cross-provider keys."""
        settings = build_model_settings(model_id="google:gemini-3-flash-preview")
        assert isinstance(settings, dict)
        assert "anthropic_cache_instructions" not in settings
        assert "openrouter_provider" not in settings

    def test_google_vertex_detected(self):
        settings = build_model_settings(model_id="google-vertex:gemini-3-flash")
        assert isinstance(settings, dict)
        assert settings["temperature"] == 0.1

    def test_google_gla_prefix_with_thinking(self):
        """google: prefix should hit the Google branch with thinking config."""
        settings = build_model_settings(
            model_id="google:gemini-3-flash-preview", thinking_budget=5000
        )
        assert isinstance(settings, dict)
        assert settings["google_thinking_config"] == {
            "include_thoughts": True,
            "thinking_budget": 5000,
        }
