"""Subscription-only routing must reject API fallback even with API keys present."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from assistant_runtime.services.llm._codex_model import OpenAICodexResponsesModel
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.exceptions import ProviderConfigError
from assistant_runtime.services.llm.interface import LlmService


@pytest.mark.parametrize("path", ["build_agent", "execute"])
@pytest.mark.parametrize(
    "scenario", ["missing", "disconnected", "expired", "excluded", "other_provider"]
)
async def test_rejects_before_agent_or_api_transport(path, scenario, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-must-not-be-used")
    service = LlmService(LLMConfig(codex_only=True, codex_models=["gpt-5.6-sol"]))
    if scenario != "missing":
        session = (
            None
            if scenario in {"disconnected", "expired"}
            else SimpleNamespace(access_token="subscription-token", account_id="test-account")
        )
        service.set_oauth_service(SimpleNamespace(get_codex_session=lambda: session))
    model = {
        "excluded": "openai:gpt-5.6-luna",
        "other_provider": "anthropic:claude-opus-5",
    }.get(scenario, "openai:gpt-5.6-sol")
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
    service = LlmService(LLMConfig(codex_only=True))
    await service.start()
    try:
        health = await service.health_check()
        assert health["codex_only"] is True
        assert health["healthy"] is False
        assert health["providers"] == []
        # An invalid strict-mode default must fail on use, never silently select an API model.
        assert service.effective_primary_model() == "anthropic:claude-opus-5"
    finally:
        await service.stop()
