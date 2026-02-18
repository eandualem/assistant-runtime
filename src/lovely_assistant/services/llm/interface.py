"""LlmService — provider-aware Pydantic AI agent factory.

Public facade for the LLM module. Creates configured agents, executes standalone
LLM calls, and manages provider lifecycle (API key loading and export).
"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent

from lovely_assistant.services.llm._settings import build_model_settings, validate_model_id
from lovely_assistant.services.llm.config import _PROVIDER_ENV_VAR_MAP, LLMConfig, ProviderConfig
from lovely_assistant.services.llm.exceptions import LLMCallError, ProviderConfigError


class LLMResult(BaseModel):
    """Result of a standalone LLM call."""

    content: str
    model: str


class LlmService:
    """Provider-aware Pydantic AI agent factory. Implements LifecycleAware."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._providers: list[ProviderConfig] = []
        self._started = False

    async def start(self) -> None:
        """Load providers from env/JSON, export keys for Pydantic AI auto-detection."""
        providers: list[ProviderConfig] = []

        # 1. Parse providers_json if set
        if self._config.providers_json:
            try:
                providers_data = json.loads(self._config.providers_json)
                providers = [ProviderConfig(**p) for p in providers_data]
                logger.info(
                    "Loaded providers from JSON config",
                    count=len(providers),
                    providers=[p.provider for p in providers],
                )
            except (json.JSONDecodeError, ValueError) as e:
                raise ProviderConfigError(f"Invalid providers_json: {e}") from e

        # 2. Auto-detect from individual env vars (supplement, don't overwrite)
        existing = {p.provider for p in providers}
        for provider_name, env_var in _PROVIDER_ENV_VAR_MAP.items():
            if provider_name not in existing:
                api_key = os.getenv(env_var)
                if api_key:
                    from pydantic import SecretStr

                    providers.append(
                        ProviderConfig(provider=provider_name, api_key=SecretStr(api_key))
                    )

        # 3. Export keys for Pydantic AI auto-detection
        for provider in providers:
            env_var = _PROVIDER_ENV_VAR_MAP.get(provider.provider)
            if env_var and not os.getenv(env_var):
                os.environ[env_var] = provider.api_key.get_secret_value()
                logger.debug("Exported API key for Pydantic AI", provider=provider.provider)

        self._providers = providers
        self._started = True

        logger.info(
            "LLM service started",
            providers=[p.provider for p in self._providers],
            primary_model=self._config.primary_model,
        )

    async def stop(self) -> None:
        """Shutdown the LLM service."""
        self._started = False
        logger.info("LLM service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        return {
            "healthy": self._started and len(self._providers) > 0,
            "providers": [p.provider for p in self._providers],
            "primary_model": self._config.primary_model,
        }

    def build_agent(
        self,
        *,
        model: str | None = None,
        system_prompt: str,
        deps_type: type = type(None),
        toolsets: list | None = None,
        output_type: type | list[type] = str,
        thinking_budget: int | None = None,
    ) -> Agent:
        """Create a configured Pydantic AI Agent.

        Tools arrive as parameters — the assistant module passes per-request tool sets
        based on frontend machine state.

        Args:
            model: Model identifier override. Defaults to config.primary_model.
            system_prompt: System prompt instructions.
            deps_type: Agent dependencies type.
            toolsets: Optional list of toolsets to register.
            output_type: Expected output type(s).
            thinking_budget: Optional thinking token budget for extended thinking.

        Returns:
            Configured Pydantic AI Agent instance.
        """
        resolved_model = validate_model_id(model or self._config.primary_model)

        settings = build_model_settings(
            model_id=resolved_model,
            thinking_budget=thinking_budget,
        )

        agent_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "deps_type": deps_type,
            "instructions": system_prompt,
            "model_settings": settings,
            "output_type": output_type,
        }
        if toolsets:
            agent_kwargs["toolsets"] = toolsets

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
        """Execute a standalone one-shot LLM call.

        Creates a one-shot Pydantic AI Agent (no tools, no history) and runs it.

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
            LLMCallError: If the LLM call fails.
            ProviderConfigError: If the model ID is invalid.
        """
        resolved_model = validate_model_id(model or self._config.primary_model)

        logger.info(
            "Executing standalone LLM call",
            model=resolved_model,
            thinking_budget=thinking_budget,
        )

        try:
            agent = Agent(
                model=resolved_model,
                instructions=system_prompt,
            )

            model_settings = build_model_settings(
                model_id=resolved_model,
                thinking_budget=thinking_budget,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = await agent.run(user_prompt, model_settings=model_settings)

            logger.info("Standalone LLM call completed", model=resolved_model)

            return LLMResult(content=result.output, model=resolved_model)

        except ProviderConfigError:
            raise
        except Exception as e:
            raise LLMCallError(f"LLM call failed: {e}") from e
