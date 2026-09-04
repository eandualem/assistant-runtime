"""Tests for LlmService lifecycle, build_agent, and execute_llm_call."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import Fernet

from assistant_runtime.services.llm._codex_model import OpenAICodexResponsesModel
from assistant_runtime.services.llm.config import LLMConfig
from assistant_runtime.services.llm.exceptions import LLMCallError, ProviderConfigError
from assistant_runtime.services.llm.interface import LLMResult, LlmService


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
        assert health["primary_model"] == "anthropic:claude-opus-5"

    async def test_primary_model_falls_back_to_a_configured_provider(self, service, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        await service.start()
        assert service.effective_primary_model() == "openai:gpt-5.6-terra"
        assert service.resolve_model() == "openai:gpt-5.6-terra"
        assert service.effective_summarization_model() == "openai:gpt-5.6-luna"
        assert service.resolve_summarization_model() == "openai:gpt-5.6-luna"
        health = await service.health_check()
        assert health["primary_model"] == "openai:gpt-5.6-terra"

    async def test_primary_model_kept_when_its_provider_is_configured(self, service, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        await service.start()
        assert service.effective_primary_model() == "anthropic:claude-opus-5"
        assert service.resolve_summarization_model() == "anthropic:claude-haiku-4-5"

    async def test_explicit_model_wins_over_fallback(self, service, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        await service.start()
        assert service.resolve_model("openai:gpt-5.4") == "openai:gpt-5.4"

    async def test_no_providers_keeps_configured_model(self, service, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        await service.start()
        assert service.effective_primary_model() == "anthropic:claude-opus-5"

    async def test_health_check_counts_codex_oauth_as_openai_provider(self, service, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service.set_oauth_service(
            SimpleNamespace(
                get_codex_session=lambda: SimpleNamespace(
                    access_token="access-token",
                    account_id="acct_123",
                )
            )
        )
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True
        assert "openai" in health["providers"]

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

    def test_build_agent_uses_codex_model_for_openai_when_oauth_connected(self, service):
        service.set_oauth_service(
            SimpleNamespace(
                get_codex_session=lambda: SimpleNamespace(
                    access_token="access-token",
                    account_id="acct_123",
                )
            )
        )

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            service.build_agent(model="openai:gpt-5.4", system_prompt="Test")

        model_arg = mock_agent_cls.call_args.kwargs["model"]
        settings_arg = mock_agent_cls.call_args.kwargs["model_settings"]
        assert isinstance(model_arg, OpenAICodexResponsesModel)
        assert settings_arg["openai_store"] is False
        assert settings_arg["openai_send_reasoning_ids"] is False
        assert "openai_previous_response_id" not in settings_arg

    def test_build_agent_preserves_openai_reasoning_settings_for_codex(self, service):
        service.set_oauth_service(
            SimpleNamespace(
                get_codex_session=lambda: SimpleNamespace(
                    access_token="access-token",
                    account_id="acct_123",
                )
            )
        )

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_cls.return_value = MagicMock()
            service.build_agent(
                model="openai:gpt-5.4",
                system_prompt="Test",
                thinking_budget=5_000,
            )

        settings_arg = mock_agent_cls.call_args.kwargs["model_settings"]
        assert settings_arg["openai_store"] is False
        assert settings_arg["openai_send_reasoning_ids"] is False
        assert "openai_previous_response_id" not in settings_arg
        assert settings_arg["openai_reasoning_effort"] == "medium"
        assert settings_arg["openai_reasoning_summary"] == "detailed"

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

    def test_build_agent_rejects_invalid_model_format(self, service):
        """Non-lowercase model IDs are rejected with a clear error."""
        with pytest.raises(ProviderConfigError, match="must be lowercase"):
            service.build_agent(
                model="Anthropic/Claude-Sonnet-4-6",
                system_prompt="Test",
            )


class TestExecuteLlmCall:
    """LlmService.execute_llm_call tests — all mocked, no live API."""

    @pytest.fixture
    def service(self):
        return LlmService(config=LLMConfig())

    async def test_execute_returns_llm_result(self, service):
        mock_result = MagicMock()
        mock_result.output = "Hello world"

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            result = await service.execute_llm_call(
                system_prompt="You are helpful.",
                user_prompt="Say hello",
            )

        assert isinstance(result, LLMResult)
        assert result.content == "Hello world"
        assert result.model == "anthropic:claude-opus-5"

    async def test_execute_with_model_override(self, service):
        mock_result = MagicMock()
        mock_result.output = "Response"

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
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
        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
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

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
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

    async def test_execute_uses_codex_model_for_supported_openai_models(self, service):
        mock_result = MagicMock()
        mock_result.output = "OK"
        service.set_oauth_service(
            SimpleNamespace(
                get_codex_session=lambda: SimpleNamespace(
                    access_token="access-token",
                    account_id="acct_123",
                )
            )
        )

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            await service.execute_llm_call(
                system_prompt="Test",
                user_prompt="Test",
                model="openai:gpt-5.4",
            )

        model_arg = mock_agent_cls.call_args.kwargs["model"]
        assert isinstance(model_arg, OpenAICodexResponsesModel)

    async def test_execute_passes_openai_reasoning_settings_to_codex_run(self, service):
        mock_result = MagicMock()
        mock_result.output = "OK"
        service.set_oauth_service(
            SimpleNamespace(
                get_codex_session=lambda: SimpleNamespace(
                    access_token="access-token",
                    account_id="acct_123",
                )
            )
        )

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(return_value=mock_result)
            mock_agent_cls.return_value = mock_agent_instance

            await service.execute_llm_call(
                system_prompt="Test",
                user_prompt="Explain briefly.",
                model="openai:gpt-5.4",
                thinking_budget=5_000,
            )

        call_kwargs = mock_agent_instance.run.call_args.kwargs
        assert call_kwargs["model_settings"]["openai_store"] is False
        assert call_kwargs["model_settings"]["openai_send_reasoning_ids"] is False
        assert "openai_previous_response_id" not in call_kwargs["model_settings"]
        assert call_kwargs["model_settings"]["openai_reasoning_effort"] == "medium"
        assert call_kwargs["model_settings"]["openai_reasoning_summary"] == "detailed"


class TestExecuteLlmCallRetry:
    """Tests for retry behavior in execute_llm_call."""

    @pytest.fixture
    def service(self):
        return LlmService(config=LLMConfig())

    async def test_retries_on_connection_error_then_succeeds(self, service):
        """execute_llm_call retries on ConnectionError and succeeds."""
        call_count = 0
        mock_result = MagicMock()
        mock_result.output = "recovered"

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()

            async def _run_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise ConnectionError("transient")
                return mock_result

            mock_agent_instance.run = _run_side_effect
            mock_agent_cls.return_value = mock_agent_instance

            # Patch wait to avoid real sleeps
            with patch(
                "assistant_runtime.base.resilience.wait_random_exponential",
                return_value=MagicMock(return_value=0),
            ):
                result = await service.execute_llm_call(
                    system_prompt="Test",
                    user_prompt="Test",
                )

        assert result.content == "recovered"
        assert call_count == 2

    async def test_gives_up_after_max_retries(self, service):
        """execute_llm_call raises LLMCallError after exhausting retries."""
        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(side_effect=ConnectionError("permanent"))
            mock_agent_cls.return_value = mock_agent_instance

            with pytest.raises(LLMCallError) as exc_info:
                await service.execute_llm_call(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            assert exc_info.value.error_category == "CONNECTION_ERROR"
            assert exc_info.value.is_retryable is True

    async def test_does_not_retry_on_non_retryable_error(self, service):
        """Non-retryable errors (like RuntimeError) are not retried."""
        call_count = 0

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()

            async def _run_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise RuntimeError("not retryable")

            mock_agent_instance.run = _run_side_effect
            mock_agent_cls.return_value = mock_agent_instance

            with pytest.raises(LLMCallError) as exc_info:
                await service.execute_llm_call(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            assert exc_info.value.error_category == "UNKNOWN"
            assert exc_info.value.is_retryable is False

        # Should be called only once — no retries
        assert call_count == 1

    async def test_error_classified_via_classify_llm_error(self, service):
        """Errors are classified with error_category via classify_llm_error."""

        class RateLimitError(Exception):
            pass

        with patch("assistant_runtime.services.llm.interface.Agent") as mock_agent_cls:
            mock_agent_instance = MagicMock()
            mock_agent_instance.run = AsyncMock(side_effect=RateLimitError("slow down"))
            mock_agent_cls.return_value = mock_agent_instance

            with pytest.raises(LLMCallError) as exc_info:
                await service.execute_llm_call(
                    system_prompt="Test",
                    user_prompt="Test",
                )

            assert exc_info.value.error_category == "RATE_LIMIT"


class TestDbProviderKeys:
    """Tests for database-sourced provider API keys — load, reload, remove, status."""

    @pytest.fixture
    def fernet(self):
        return Fernet(Fernet.generate_key())

    @pytest.fixture
    def mock_db(self):
        """Mock database service with an async session context manager."""
        from contextlib import asynccontextmanager

        db = MagicMock()
        db._healthy = True
        session = AsyncMock()
        session.commit = AsyncMock()

        @asynccontextmanager
        async def fake_session_context():
            yield session

        db.session_context = fake_session_context
        db._mock_session = session
        return db

    def _make_service_with_db(self, mock_db, fernet):
        """Create an LlmService wired to mock DB + encryption."""
        config = LLMConfig()
        service = LlmService(config=config)
        service._db_service = mock_db
        service._fernet = fernet
        return service

    async def test_start_loads_keys_from_database(self, mock_db, fernet, monkeypatch):
        """When DB has an encrypted key, start() loads it and exports to env."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        encrypted = fernet.encrypt(b"sk-from-database").decode()
        mock_token = SimpleNamespace(encrypted_api_key=encrypted)

        async def fake_get(provider):
            if provider == "anthropic":
                return mock_token
            return None

        with patch(
            "assistant_runtime.services.database.repositories.OAuthTokenRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get = AsyncMock(side_effect=fake_get)
            mock_repo_cls.return_value = mock_repo

            service = self._make_service_with_db(mock_db, fernet)
            await service.start()

        assert len(service._providers) == 1
        assert service._providers[0].provider == "anthropic"
        assert service._providers[0].api_key.get_secret_value() == "sk-from-database"
        assert service._db_providers.get("anthropic") == "database"

        import os

        assert os.getenv("ANTHROPIC_API_KEY") == "sk-from-database"

    async def test_db_keys_take_priority_over_env(self, mock_db, fernet, monkeypatch):
        """DB key is used even when the env var already exists."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        encrypted = fernet.encrypt(b"sk-from-db").decode()
        mock_token = SimpleNamespace(encrypted_api_key=encrypted)

        async def fake_get(provider):
            if provider == "anthropic":
                return mock_token
            return None

        with patch(
            "assistant_runtime.services.database.repositories.OAuthTokenRepository"
        ) as mock_repo_cls:
            mock_repo = MagicMock()
            mock_repo.get = AsyncMock(side_effect=fake_get)
            mock_repo_cls.return_value = mock_repo

            service = self._make_service_with_db(mock_db, fernet)
            await service.start()

        # DB key should be in the provider list (not the env-var one)
        anthropic_providers = [p for p in service._providers if p.provider == "anthropic"]
        assert len(anthropic_providers) == 1
        assert anthropic_providers[0].api_key.get_secret_value() == "sk-from-db"

        # DB loads first, and env auto-detect skips already-existing providers
        assert service._db_providers.get("anthropic") == "database"

    async def test_reload_provider_key_updates_providers_and_env(self, monkeypatch):
        """reload_provider_key hot-reloads the key in memory and env."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        service = LlmService(config=LLMConfig())
        await service.start()

        # Initially no providers
        assert len(service._providers) == 0

        # Reload a key
        await service.reload_provider_key("anthropic", "sk-new-key")

        assert len(service._providers) == 1
        assert service._providers[0].provider == "anthropic"
        assert service._providers[0].api_key.get_secret_value() == "sk-new-key"
        assert service._db_providers["anthropic"] == "database"

        import os

        assert os.getenv("ANTHROPIC_API_KEY") == "sk-new-key"

    async def test_remove_provider_key_clears_providers_and_env(self, monkeypatch):
        """remove_provider_key removes from in-memory providers and env."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-to-remove")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        service = LlmService(config=LLMConfig())
        await service.start()

        # Should have anthropic loaded from env
        assert any(p.provider == "anthropic" for p in service._providers)

        # Remove it
        await service.remove_provider_key("anthropic")

        assert not any(p.provider == "anthropic" for p in service._providers)
        assert "anthropic" not in service._db_providers

        import os

        assert os.getenv("ANTHROPIC_API_KEY") is None

    async def test_get_provider_status_returns_all_providers(self, monkeypatch):
        """get_provider_status includes entries for all four known providers."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        service = LlmService(config=LLMConfig())
        await service.start()

        statuses = service.get_provider_status()
        provider_names = {s["provider"] for s in statuses}
        assert provider_names == {"anthropic", "openai", "google", "openrouter"}

        # All should be unconfigured
        for status in statuses:
            assert status["configured"] is False
            assert status["source"] is None

    async def test_get_provider_status_shows_db_source(self, monkeypatch):
        """DB-loaded keys show source='database' in provider status."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        service = LlmService(config=LLMConfig())
        await service.start()

        # Simulate a DB-loaded key via reload
        await service.reload_provider_key("anthropic", "sk-db-key-5678")

        statuses = service.get_provider_status()
        by_provider = {s["provider"]: s for s in statuses}

        anthropic = by_provider["anthropic"]
        assert anthropic["configured"] is True
        assert anthropic["source"] == "database"
        assert anthropic["api_key_preview"] == "...5678"

        # Other providers should remain unconfigured
        assert by_provider["openai"]["configured"] is False

    async def test_get_provider_status_shows_env_source(self, monkeypatch):
        """Env-loaded keys show source='environment' in provider status."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-abcd")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        service = LlmService(config=LLMConfig())
        await service.start()

        statuses = service.get_provider_status()
        by_provider = {s["provider"]: s for s in statuses}

        anthropic = by_provider["anthropic"]
        assert anthropic["configured"] is True
        assert anthropic["source"] == "environment"
        assert anthropic["api_key_preview"] == "...abcd"
