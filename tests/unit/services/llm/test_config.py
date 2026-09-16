"""Tests for LLM service configuration models."""

import pytest
from pydantic import SecretStr, ValidationError

from assistant_runtime.model_catalog import ALLOWED_PROVIDERS
from assistant_runtime.services.llm.config import LLMConfig, ProviderConfig


class TestProviderConfig:
    """ProviderConfig validation tests."""

    def test_valid_anthropic_provider(self):
        config = ProviderConfig(provider="anthropic", api_key=SecretStr("sk-test"))
        assert config.provider == "anthropic"
        assert config.api_key.get_secret_value() == "sk-test"

    def test_valid_openrouter_provider(self):
        config = ProviderConfig(provider="openrouter", api_key=SecretStr("sk-or-test"))
        assert config.provider == "openrouter"

    def test_transport_fields_are_no_longer_accepted(self):
        """base_url, timeout and max_retries were accepted and ignored; now they are refused."""
        for field in ({"base_url": "https://x"}, {"timeout": 5}, {"max_retries": 1}):
            with pytest.raises(ValidationError):
                ProviderConfig(provider="anthropic", api_key=SecretStr("key"), **field)

    def test_all_allowed_providers(self):
        for provider in ALLOWED_PROVIDERS:
            config = ProviderConfig(provider=provider, api_key=SecretStr("key"))
            assert config.provider == provider

    def test_invalid_provider_rejected(self):
        with pytest.raises(ValidationError, match="Provider must be one of"):
            ProviderConfig(provider="bedrock", api_key=SecretStr("key"))

    def test_missing_api_key_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(provider="anthropic")

    def test_frozen(self):
        config = ProviderConfig(provider="anthropic", api_key=SecretStr("sk-test"))
        with pytest.raises(ValidationError):
            config.provider = "openai"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ProviderConfig(provider="anthropic", api_key=SecretStr("key"), unknown_field="value")


class TestLLMConfig:
    """LLMConfig defaults and validation tests."""

    def test_defaults(self):
        config = LLMConfig()
        assert config.primary_model == "anthropic:claude-opus-5"
        assert config.summarization_model == "anthropic:claude-haiku-4-5"
        assert config.providers_json is None

    def test_custom_models(self):
        config = LLMConfig(
            primary_model="openai:gpt-4o",
            summarization_model="openai:gpt-4o-mini",
        )
        assert config.primary_model == "openai:gpt-4o"
        assert config.summarization_model == "openai:gpt-4o-mini"

    def test_providers_json_field(self):
        config = LLMConfig(providers_json='[{"provider": "anthropic", "api_key": "sk-test"}]')
        assert config.providers_json is not None

    def test_frozen(self):
        config = LLMConfig()
        with pytest.raises(ValidationError):
            config.primary_model = "something"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            LLMConfig(unknown="value")
