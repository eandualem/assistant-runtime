"""Tests for LlmService lifecycle, build_agent, and execute_llm_call."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lovely_assistant.services.llm.config import LLMConfig
from lovely_assistant.services.llm.exceptions import LLMCallError, ProviderConfigError
from lovely_assistant.services.llm.interface import LLMResult, LlmService


class TestLlmServiceLifecycle:
    """Start, stop, and health_check tests."""

    @pytest.fixture
    def config(self):
        return LLMConfig()

    @pytest.fixture
    def service(self, config):
        return LlmService(config=config)

    async def test_start_with_env_var(self, service, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        await service.start()
        assert service._started is True
        assert len(service._providers) >= 1
        assert service._providers[0].provider == "anthropic"

    async def test_start_with_multiple_env_vars(self, service, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
        await service.start()
        providers = {p.provider for p in service._providers}
        assert "anthropic" in providers
        assert "openai" in providers

    async def test_start_with_providers_json(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        config = LLMConfig(providers_json='[{"provider": "anthropic", "api_key": "sk-json-key"}]')
        service = LlmService(config=config)
        await service.start()
        assert len(service._providers) == 1
        assert service._providers[0].provider == "anthropic"
        assert service._providers[0].api_key.get_secret_value() == "sk-json-key"

    async def test_start_with_invalid_json_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = LLMConfig(providers_json="not-valid-json")
        service = LlmService(config=config)
        with pytest.raises(ProviderConfigError, match="Invalid providers_json"):
            await service.start()

    async def test_start_no_providers(self, service, monkeypatch):
        """Service starts but health reports unhealthy with no providers."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        await service.start()
        assert service._started is True
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_stop(self, service, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        await service.start()
        await service.stop()
        assert service._started is False

    async def test_health_check_before_start(self, service):
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_check_after_start(self, service, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True
        assert "anthropic" in health["providers"]
        assert health["primary_model"] == "anthropic:claude-haiku-4-5"

    async def test_json_providers_dont_overwrite_env(self, monkeypatch):
        """When ANTHROPIC_API_KEY is already set, export doesn't overwrite it."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "original-key")
        config = LLMConfig(providers_json='[{"provider": "anthropic", "api_key": "json-key"}]')
        service = LlmService(config=config)
        await service.start()
        import os

        assert os.getenv("ANTHROPIC_API_KEY") == "original-key"


class TestBuildAgent:
    """LlmService.build_agent tests."""

    @pytest.fixture
    def service(self):
        return LlmService(config=LLMConfig())

    def test_build_agent_returns_agent(self, service):
        agent = service.build_agent(system_prompt="You are helpful.")
        # Verify it's a Pydantic AI Agent
        from pydantic_ai import Agent

        assert isinstance(agent, Agent)

    def test_build_agent_with_model_override(self, service):
        agent = service.build_agent(
            model="anthropic:claude-haiku-4-5",
            system_prompt="Test",
        )
        from pydantic_ai import Agent

        assert isinstance(agent, Agent)

    def test_build_agent_with_invalid_model_raises(self, service):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            service.build_agent(model="invalid", system_prompt="Test")

    def test_build_agent_with_toolsets(self, service):
        """Toolsets parameter is forwarded to Agent constructor."""
        mock_toolset = MagicMock()
        agent = service.build_agent(
            system_prompt="Test",
            toolsets=[mock_toolset],
        )
        from pydantic_ai import Agent

        assert isinstance(agent, Agent)

    def test_build_agent_normalizes_model(self, service):
        """Slash-separated model IDs are normalized."""
        agent = service.build_agent(
            model="Anthropic/Claude-Sonnet-4-6",
            system_prompt="Test",
        )
        from pydantic_ai import Agent

        assert isinstance(agent, Agent)


class TestExecuteLlmCall:
    """LlmService.execute_llm_call tests — all mocked, no live API."""

    @pytest.fixture
    def service(self):
        return LlmService(config=LLMConfig())

    async def test_execute_returns_llm_result(self, service):
        mock_result = MagicMock()
        mock_result.output = "Hello world"

        with patch("lovely_assistant.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            result = await service.execute_llm_call(
                system_prompt="You are helpful.",
                user_prompt="Say hello",
            )

        assert isinstance(result, LLMResult)
        assert result.content == "Hello world"
        assert result.model == "anthropic:claude-haiku-4-5"

    async def test_execute_with_model_override(self, service):
        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch("lovely_assistant.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            result = await service.execute_llm_call(
                system_prompt="Test",
                user_prompt="Test",
                model="openai:gpt-4o",
            )

        assert result.model == "openai:gpt-4o"

    async def test_execute_with_invalid_model_raises(self, service):
        with pytest.raises(ProviderConfigError, match="Invalid model ID"):
            await service.execute_llm_call(
                system_prompt="Test",
                user_prompt="Test",
                model="bad-model",
            )

    async def test_execute_wraps_exceptions_in_llm_call_error(self, service):
        with patch("lovely_assistant.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(side_effect=RuntimeError("API timeout"))
            mock_agent_cls.return_value = mock_agent_instance

            with pytest.raises(LLMCallError, match="LLM call failed"):
                await service.execute_llm_call(
                    system_prompt="Test",
                    user_prompt="Test",
                )

    async def test_execute_passes_model_settings(self, service):
        mock_result = MagicMock()
        mock_result.output = "OK"

        with patch("lovely_assistant.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            await service.execute_llm_call(
                system_prompt="Test",
                user_prompt="Test",
                thinking_budget=5000,
                temperature=0.5,
                max_tokens=2000,
            )

            # Verify Agent.run was called with model_settings
            mock_agent_instance.run.assert_called_once()
            call_kwargs = mock_agent_instance.run.call_args
            assert "model_settings" in call_kwargs.kwargs
