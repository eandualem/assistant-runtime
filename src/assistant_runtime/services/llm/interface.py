"""LlmService — provider-aware Pydantic AI agent factory.

Public facade for the LLM module. Creates configured agents, executes standalone
LLM calls, and manages provider lifecycle (API key loading and export).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider

from assistant_runtime.base.resilience import retry_with_backoff
from assistant_runtime.services.llm._codex_model import OpenAICodexResponsesModel
from assistant_runtime.services.llm._settings import build_model_settings, validate_model_id
from assistant_runtime.services.llm.config import (
    PROVIDER_DEFAULT_MODELS,
    PROVIDER_DEFAULT_SUMMARIZATION_MODELS,
    PROVIDER_ENV_VARS,
    LLMConfig,
    ProviderConfig,
)
from assistant_runtime.services.llm.exceptions import ProviderConfigError, classify_llm_error
from assistant_runtime.services.tracing import create_span


def _collect_retryable_llm_exceptions() -> tuple[type[Exception], ...]:
    """Collect retryable exception types from installed LLM provider SDKs.

    Lazily imports from anthropic, openai, and google-genai to avoid hard
    dependency on packages that may not be installed.
    """
    types: list[type[Exception]] = [ConnectionError, TimeoutError]

    for module_path in ("anthropic", "openai"):
        try:
            mod = __import__(module_path)
            for name in (
                "RateLimitError",
                "InternalServerError",
                "APIConnectionError",
                "APITimeoutError",
            ):
                cls = getattr(mod, name, None)
                if cls is not None:
                    types.append(cls)
        except ImportError:
            pass

    # google-genai uses a different module path
    try:
        from google.genai import errors as google_errors

        for name in ("ClientError", "ServerError"):
            cls = getattr(google_errors, name, None)
            if cls is not None:
                types.append(cls)
    except ImportError:
        pass

    return tuple(types)


_LLM_RETRYABLE_EXCEPTIONS = _collect_retryable_llm_exceptions()
_CODEX_BACKEND_BASE_URL = "https://chatgpt.com/backend-api/codex/"
_CODEX_MODEL_NAMES = frozenset({"gpt-5.4", "gpt-5.4-pro"})


class LLMResult(BaseModel):
    """Result of a standalone LLM call."""

    content: str
    model: str


class LlmService:
    """Provider-aware Pydantic AI agent factory. Implements LifecycleAware."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._providers: list[ProviderConfig] = []
        self._oauth_service: Any | None = None
        self._db_service: Any | None = None
        self._fernet: Any | None = None
        self._db_providers: dict[str, str] = {}  # provider → "database" (tracks DB-sourced keys)
        self._codex_provider: OpenAIProvider | None = None
        self._codex_provider_identity: tuple[str, str] | None = None
        self._codex_http_clients: list[httpx.AsyncClient] = []
        self._started = False

    def set_oauth_service(self, oauth_service: Any) -> None:
        """Attach OAuth service after lifecycle registration."""
        self._oauth_service = oauth_service

    def set_database_service(self, db_service: Any, encryption_key: str = "") -> None:
        """Attach database service and encryption key for DB-stored API keys."""
        self._db_service = db_service
        if encryption_key:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(encryption_key.encode())

    async def start(self) -> None:
        """Load providers from DB/env/JSON, export keys for Pydantic AI auto-detection."""
        from pydantic import SecretStr

        providers: list[ProviderConfig] = []

        # 0. Load API keys from database (highest priority — user explicitly configured)
        if self._db_service is not None and self._fernet is not None:
            db_providers = await self._load_db_provider_keys()
            providers.extend(db_providers)

        # 1. Parse providers_json if set (supplement, don't overwrite DB keys)
        existing = {p.provider for p in providers}
        if self._config.providers_json:
            try:
                providers_data = json.loads(self._config.providers_json)
                for p_data in providers_data:
                    pc = ProviderConfig(**p_data)
                    if pc.provider not in existing:
                        providers.append(pc)
                        existing.add(pc.provider)
                logger.info(
                    "Loaded providers from JSON config",
                    count=len(providers),
                    providers=[p.provider for p in providers],
                )
            except (json.JSONDecodeError, ValueError) as e:
                raise ProviderConfigError(f"Invalid providers_json: {e}") from e

        # 2. Auto-detect from individual env vars (supplement, don't overwrite)
        for provider_name, env_var in PROVIDER_ENV_VARS.items():
            if provider_name not in existing:
                api_key = os.getenv(env_var)
                if api_key:
                    providers.append(
                        ProviderConfig(provider=provider_name, api_key=SecretStr(api_key))
                    )

        # 3. Export keys for Pydantic AI auto-detection
        for provider in providers:
            env_var = PROVIDER_ENV_VARS.get(provider.provider)
            if env_var and not os.getenv(env_var):
                os.environ[env_var] = provider.api_key.get_secret_value()
                logger.debug("Exported API key for Pydantic AI", provider=provider.provider)

        self._providers = providers
        self._started = True

        configured_providers = self._configured_provider_names()
        if not configured_providers:
            logger.warning(
                "No LLM providers configured. Set ANTHROPIC_API_KEY in .env or environment."
            )

        logger.info(
            "LLM service started",
            providers=configured_providers,
            primary_model=self.effective_primary_model(),
        )

    async def stop(self) -> None:
        """Shutdown the LLM service."""
        for client in self._codex_http_clients:
            await client.aclose()
        self._codex_http_clients.clear()
        self._codex_provider = None
        self._codex_provider_identity = None
        self._started = False
        logger.info("LLM service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        providers = self._configured_provider_names()
        return {
            "healthy": self._started and len(providers) > 0,
            "providers": providers,
            "primary_model": self.effective_primary_model(),
        }

    async def reload_provider_key(self, provider: str, api_key: str) -> None:
        """Hot-reload a provider API key — updates in-memory state and env var.

        Called by the providers route after storing the key in the DB.
        """
        from pydantic import SecretStr

        if provider not in PROVIDER_ENV_VARS:
            raise ProviderConfigError(f"Unknown provider: {provider}")

        # Update or add provider in memory
        self._providers = [p for p in self._providers if p.provider != provider]
        self._providers.append(ProviderConfig(provider=provider, api_key=SecretStr(api_key)))

        # Track as DB-sourced and export to env var for Pydantic AI
        self._db_providers[provider] = "database"
        env_var = PROVIDER_ENV_VARS[provider]
        os.environ[env_var] = api_key
        logger.info("Provider API key reloaded", provider=provider)

    async def remove_provider_key(self, provider: str) -> None:
        """Remove a provider API key from in-memory state and env var.

        Called by the providers route after deleting the key from the DB.
        """
        if provider not in PROVIDER_ENV_VARS:
            raise ProviderConfigError(f"Unknown provider: {provider}")

        self._providers = [p for p in self._providers if p.provider != provider]
        self._db_providers.pop(provider, None)
        env_var = PROVIDER_ENV_VARS[provider]
        os.environ.pop(env_var, None)
        logger.info("Provider API key removed", provider=provider)

    def get_provider_status(self) -> list[dict[str, Any]]:
        """Return provider auth status for each known provider."""
        configured = {p.provider for p in self._providers}
        result = []
        for provider_name in PROVIDER_ENV_VARS:
            has_key = provider_name in configured
            # Determine source: check if from DB (tracked in _db_providers) or env
            source = None
            if has_key:
                source = self._db_providers.get(provider_name, "environment")
            api_key_preview = None
            if has_key:
                for p in self._providers:
                    if p.provider == provider_name:
                        raw = p.api_key.get_secret_value()
                        api_key_preview = f"...{raw[-4:]}" if len(raw) >= 4 else "***"
                        break
            result.append(
                {
                    "provider": provider_name,
                    "configured": has_key,
                    "source": source,
                    "api_key_preview": api_key_preview,
                }
            )
        return result

    async def _load_db_provider_keys(self) -> list[ProviderConfig]:
        """Load encrypted API keys from the oauth_tokens table."""
        from pydantic import SecretStr

        providers: list[ProviderConfig] = []
        try:
            async with self._db_service.session_context() as session:
                from assistant_runtime.services.database.repositories import OAuthTokenRepository

                repo = OAuthTokenRepository(session)
                for provider_name in PROVIDER_ENV_VARS:
                    token = await repo.get(provider_name)
                    if token is not None and token.encrypted_api_key:
                        api_key = self._fernet.decrypt(token.encrypted_api_key.encode()).decode()
                        providers.append(
                            ProviderConfig(provider=provider_name, api_key=SecretStr(api_key))
                        )
                        self._db_providers[provider_name] = "database"
                        logger.debug("Loaded API key from database", provider=provider_name)
        except Exception as exc:
            logger.warning("Failed to load provider keys from database", error=str(exc))
        return providers

    def resolve_model(self, model: str | None = None) -> str:
        """Resolve, normalize, and validate a model identifier."""
        return validate_model_id(model or self.effective_primary_model())

    def resolve_summarization_model(self, model: str | None = None) -> str:
        """The model for summaries and other lightweight tasks, validated."""
        return validate_model_id(model or self.effective_summarization_model())

    def effective_primary_model(self) -> str:
        """The configured primary model, or a configured provider's default.

        With only an OpenAI key set, the default ``anthropic:claude-opus-5``
        cannot be used; the first configured provider's default is used
        instead (and logged) so a one-key setup works out of the box.
        """
        return self._effective_model(self._config.primary_model, PROVIDER_DEFAULT_MODELS)

    def effective_summarization_model(self) -> str:
        """The configured summarization model, or a configured provider's default."""
        return self._effective_model(
            self._config.summarization_model, PROVIDER_DEFAULT_SUMMARIZATION_MODELS
        )

    def _effective_model(self, configured: str, defaults: dict[str, str]) -> str:
        provider = configured.split(":", 1)[0]
        available = self._configured_provider_names() if self._started else []
        if not available:
            return configured
        if provider in available or (provider == "google-cloud" and "google" in available):
            return configured
        fallback = next((defaults[p] for p in defaults if p in available), None)
        if fallback is None:
            return configured
        logger.warning(
            "Configured model's provider has no credentials; using a configured provider's default",
            configured=configured,
            fallback=fallback,
            providers=available,
        )
        return fallback

    def _configured_provider_names(self) -> list[str]:
        """Return provider names including active Codex-backed OpenAI auth."""
        providers = {p.provider for p in self._providers}
        if self._get_codex_session() is not None:
            providers.add("openai")
        return sorted(providers)

    def _get_codex_session(self) -> Any | None:
        """Get the active Codex session from OAuth, if available."""
        if self._oauth_service is None:
            return None

        getter = getattr(self._oauth_service, "get_codex_session", None)
        if getter is None:
            return None

        return getter()

    def _should_use_codex_provider(self, resolved_model: str) -> bool:
        """Whether this model should use ChatGPT/Codex subscription auth."""
        if not resolved_model.startswith("openai:"):
            return False
        model_name = resolved_model.split(":", 1)[1]
        return model_name in _CODEX_MODEL_NAMES and self._get_codex_session() is not None

    def _apply_model_transport_defaults(
        self,
        resolved_model: str,
        settings: dict[str, Any] | Any,
    ) -> dict[str, Any] | Any:
        """Apply transport-specific defaults for certain providers."""
        if not self._should_use_codex_provider(resolved_model):
            return settings

        codex_settings: dict[str, Any] = {"openai_store": False}
        if isinstance(settings, dict):
            for key in (
                "timeout",
                "extra_headers",
                "extra_body",
                "openai_reasoning_effort",
                "openai_reasoning_summary",
                "openai_send_reasoning_ids",
                "openai_truncation",
                "openai_user",
            ):
                if key in settings:
                    codex_settings[key] = settings[key]
        return codex_settings

    def _get_or_create_codex_provider(self, session: Any) -> OpenAIProvider:
        """Reuse a custom OpenAI provider for the current Codex access token."""
        identity = (session.access_token, session.account_id)
        if self._codex_provider is not None and self._codex_provider_identity == identity:
            return self._codex_provider

        http_client = httpx.AsyncClient(timeout=120.0)
        self._codex_http_clients.append(http_client)
        openai_client = AsyncOpenAI(
            api_key=session.access_token,
            base_url=_CODEX_BACKEND_BASE_URL,
            default_headers={"ChatGPT-Account-Id": session.account_id},
            http_client=http_client,
        )
        self._codex_provider = OpenAIProvider(openai_client=openai_client)
        self._codex_provider_identity = identity
        return self._codex_provider

    def _resolve_agent_model(self, resolved_model: str) -> str | OpenAICodexResponsesModel:
        """Resolve the actual Agent model object to use for a model id."""
        if not self._should_use_codex_provider(resolved_model):
            return resolved_model

        session = self._get_codex_session()
        assert session is not None
        model_name = resolved_model.split(":", 1)[1]
        provider = self._get_or_create_codex_provider(session)
        return OpenAICodexResponsesModel(model_name=model_name, provider=provider)

    def build_agent(
        self,
        *,
        model: str | None = None,
        system_prompt: str,
        deps_type: type = type(None),
        toolsets: list | None = None,
        output_type: type | list[type] = str,
        thinking_budget: int | None = None,
        temperature: float | None = None,
    ) -> Agent:
        """Create a configured Pydantic AI Agent.

        Tools arrive as parameters — The assistant module passes per-request tool sets
        based on the host context.

        Args:
            model: Model identifier override. Defaults to the effective primary model.
            system_prompt: System prompt instructions.
            deps_type: Agent dependencies type.
            toolsets: Optional list of toolsets to register.
            output_type: Expected output type(s).
            thinking_budget: Optional thinking token budget for extended thinking.
            temperature: Optional temperature override.

        Returns:
            Configured Pydantic AI Agent instance.
        """
        resolved_model = self.resolve_model(model)

        settings = build_model_settings(
            model_id=resolved_model,
            thinking_budget=thinking_budget,
            temperature=temperature,
        )
        settings = self._apply_model_transport_defaults(resolved_model, settings)
        agent_model = self._resolve_agent_model(resolved_model)

        agent_kwargs: dict[str, Any] = {
            "model": agent_model,
            "deps_type": deps_type,
            "instructions": system_prompt,
            "model_settings": settings,
            "output_type": output_type,
        }
        if toolsets:
            agent_kwargs["toolsets"] = toolsets
            for ts in toolsets:
                ts_name = getattr(ts, "id", None) or type(ts).__name__
                logger.info("Agent toolset", name=ts_name, type=type(ts).__name__)

        agent = Agent(**agent_kwargs)

        logger.info(
            "Built Pydantic AI agent",
            model=resolved_model,
            toolsets=len(toolsets) if toolsets else 0,
            thinking="enabled" if thinking_budget else "disabled",
        )

        return agent

    async def execute_llm_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        thinking_budget: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult:
        """Execute a standalone one-shot LLM call with retry on transient failures.

        Creates a one-shot Pydantic AI Agent (no tools, no history) and runs it.
        Retries up to 3 times on rate limits, server errors, and connection failures.

        Args:
            system_prompt: System instructions for the LLM.
            user_prompt: User message to send.
            model: Model identifier override.
            thinking_budget: Optional thinking token budget.
            temperature: Temperature override.
            max_tokens: Max tokens override.

        Returns:
            LLMResult with content and model identifier.

        Raises:
            LLMCallError: If the LLM call fails after all retries.
            ProviderConfigError: If the model ID is invalid.
        """
        resolved_model = self.resolve_model(model)

        logger.info(
            "Executing standalone LLM call",
            model=resolved_model,
            thinking_budget=thinking_budget,
        )

        @retry_with_backoff(
            max_attempts=3,
            min_wait=1.0,
            max_wait=30.0,
            retry_on=_LLM_RETRYABLE_EXCEPTIONS,
            name="execute_llm_call",
        )
        async def _run_with_retry() -> LLMResult:
            agent_model = self._resolve_agent_model(resolved_model)
            agent = Agent(
                model=agent_model,
                instructions=system_prompt,
            )

            model_settings = build_model_settings(
                model_id=resolved_model,
                thinking_budget=thinking_budget,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            model_settings = self._apply_model_transport_defaults(resolved_model, model_settings)

            result = await agent.run(user_prompt, model_settings=model_settings)

            logger.info("Standalone LLM call completed", model=resolved_model)

            return LLMResult(content=result.output, model=resolved_model)

        with create_span(
            "standalone-llm-call",
            input_data=user_prompt,
            metadata={"model": resolved_model},
        ) as span:
            try:
                llm_result = await _run_with_retry()
                span.update_output(llm_result.content)
                return llm_result
            except ProviderConfigError:
                raise
            except Exception as e:
                raise classify_llm_error(e) from e
