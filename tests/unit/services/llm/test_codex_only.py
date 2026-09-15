"""Subscription-only routing must reject API fallback even with API keys present."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from assistant_runtime.services.llm._codex_model import OpenAICodexResponsesModel
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.exceptions import ProviderConfigError
from assistant_runtime.services.llm.interface import LlmService


@pytest.mark.parametrize("tier", [None, "fast"])
@pytest.mark.parametrize("path", ["build_agent", "execute"])
@pytest.mark.parametrize("scenario", ["missing", "disconnected", "expired", "excluded"])
async def test_rejects_before_agent_or_api_transport(path, scenario, tier, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-must-not-be-used")
    service = LlmService(
        LLMConfig(codex_only=True, codex_models=["gpt-5.6-sol"], codex_service_tier=tier)
    )
    if scenario != "missing":
        session = (
            None
            if scenario in {"disconnected", "expired"}
            else SimpleNamespace(access_token="subscription-token", account_id="test-account")
        )
        service.set_oauth_service(SimpleNamespace(get_codex_session=lambda: session))
    model = {"excluded": "openai:gpt-5.6-luna"}.get(scenario, "openai:gpt-5.6-sol")
    with (
        patch("assistant_runtime.services.llm.interface.Agent") as agent,
        patch("assistant_runtime.services.llm.interface.AsyncOpenAI") as client,
    ):
        if path == "execute":
            with pytest.raises(ProviderConfigError):
                await service.execute_llm_call(
                    model=model, system_prompt="test", user_prompt="test"
                )
        else:
            with pytest.raises(ProviderConfigError):
                service.build_agent(model=model, system_prompt="test")
        agent.assert_not_called()
        client.assert_not_called()


async def test_subscription_routes_main_and_auxiliary_models_then_blocks_disconnect(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-must-not-be-used")
    service = LlmService(
        LLMConfig(
            codex_only=True,
            primary_model="openai:gpt-5.6-sol",
            summarization_model="openai:gpt-5.6-luna",
        )
    )
    service.set_oauth_service(
        SimpleNamespace(
            get_codex_session=lambda: SimpleNamespace(
                access_token="subscription-token", account_id="test-account"
            )
        )
    )
    try:
        for model in (service.resolve_model(), service.resolve_summarization_model()):
            agent = service.build_agent(model=model, system_prompt="test", thinking_budget=4000)
            assert isinstance(agent.model, OpenAICodexResponsesModel)
            assert str(agent.model.base_url) == "https://chatgpt.com/backend-api/codex/"
            assert agent.model_settings["openai_reasoning_effort"] == "low"
            assert "temperature" not in agent.model_settings
        service.set_oauth_service(SimpleNamespace(get_codex_session=lambda: None))
        with pytest.raises(ProviderConfigError, match="API fallback is disabled"):
            service.build_agent(system_prompt="test")
    finally:
        await service.stop()


async def test_health_and_defaults_do_not_fallback_to_configured_api_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    for name in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    service = LlmService(LLMConfig(codex_only=True))
    await service.start()
    try:
        health = await service.health_check()
        assert health["codex_only"] is True
        assert health["healthy"] is False
        assert health["providers"] == []  # the API key does not make openai available
        # An invalid strict-mode default must fail on use, never silently select an API model.
        assert service.effective_primary_model() == "anthropic:claude-opus-5"
    finally:
        await service.stop()


async def test_other_providers_stay_routable_under_the_subscription_guard(monkeypatch):
    """The guard is about OpenAI billing: a provider with its own key is its own choice."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-must-not-be-used")
    monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
    for name in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    service = LlmService(LLMConfig(codex_only=True, primary_model="openai:gpt-5.6-sol"))
    await service.start()
    try:
        assert (await service.health_check())["providers"] == ["cerebras"]
        with patch("assistant_runtime.services.llm.interface.AsyncOpenAI") as client:
            agent = service.build_agent(model="cerebras:gpt-oss-120b", system_prompt="test")
            client.assert_not_called()  # no Codex transport for another provider
        assert agent.model.system == "cerebras"
        # openai: models still need the subscription; the API key is never used.
        with pytest.raises(ProviderConfigError, match="API fallback is disabled"):
            service.build_agent(model="openai:gpt-5.6-sol", system_prompt="test")
        # The configured openai default is kept, not swapped for another provider.
        assert service.effective_primary_model() == "openai:gpt-5.6-sol"
    finally:
        await service.stop()


async def test_the_turns_tier_overrides_the_startup_default(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = LlmService(LLMConfig(codex_only=True, codex_service_tier="default"))
    service.set_oauth_service(
        SimpleNamespace(
            get_codex_session=lambda: SimpleNamespace(
                access_token="subscription-token", account_id="test-account"
            )
        )
    )
    try:
        default = service.build_agent(model="openai:gpt-5.6-sol", system_prompt="t")
        fast = service.build_agent(
            model="openai:gpt-5.6-sol", system_prompt="t", codex_service_tier="fast"
        )
        assert default.model_settings["openai_service_tier"] == "default"
        assert fast.model_settings["openai_service_tier"] == "priority"
    finally:
        await service.stop()
