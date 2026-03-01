"""Internal model settings construction for Pydantic AI agents.

Translated from arclio-assistant's agent_factory.py (build_model_settings)
and agent_config.py (validation, constants).

Not part of the public module API — imported only by interface.py.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from lovely_assistant.services.llm.exceptions import ProviderConfigError

# Internal constants for computed LLM parameters
_RESPONSE_MAX_TOKENS = 8_192
_DEFAULT_TEMPERATURE = 0.1
_THINKING_TEMPERATURE = 1.0


def validate_model_id(model_id: str) -> str:
    """Validate a model identifier. Raises ProviderConfigError on invalid format.

    Model IDs must be lowercase, colon-separated (provider:model-name).
    No normalization — wrong format is an error.
    """
    if not model_id or not model_id.strip():
        raise ProviderConfigError("Model ID cannot be empty")

    model_id = model_id.strip()

    # Reject non-lowercase
    if model_id != model_id.lower():
        raise ProviderConfigError(
            f"Model ID must be lowercase: '{model_id}'. Use '{model_id.lower()}'"
        )

    # Reject slash separator (when no colon present — openrouter uses slashes after the colon)
    if "/" in model_id and ":" not in model_id:
        raise ProviderConfigError(
            f"Model ID must use colon separator: '{model_id}'. "
            f"Use '{model_id.replace('/', ':', 1)}'"
        )

    # Reject legacy google: prefix
    if model_id.startswith("google:"):
        raise ProviderConfigError(
            f"Use 'google-gla:' prefix, not 'google:': '{model_id}'. "
            f"Use 'google-gla:{model_id[len('google:'):]}'"
        )

    # Validate colon-separated format
    if ":" not in model_id:
        raise ProviderConfigError(
            f"Invalid model ID: '{model_id}'. "
            f"Must use 'provider:model-name' format (e.g., 'anthropic:claude-sonnet-4-6')."
        )
    parts = model_id.split(":", 1)
    if not parts[0] or not parts[1]:
        raise ProviderConfigError(
            f"Invalid model ID: '{model_id}'. "
            f"Must use 'provider:model-name' format (e.g., 'anthropic:claude-sonnet-4-6')."
        )

    return model_id


def build_model_settings(
    *,
    model_id: str,
    thinking_budget: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> AnthropicModelSettings | GoogleModelSettings | OpenRouterModelSettings | dict[str, Any]:
    """Build provider-specific model settings for a Pydantic AI Agent.

    Determines the correct settings type based on the model ID prefix and configures
    temperature, max_tokens, thinking, caching, and provider-specific options.

    Args:
        model_id: Full model identifier (e.g., "anthropic:claude-sonnet-4-6").
        thinking_budget: Optional thinking token budget. None = disabled.
        temperature: Optional temperature override. Ignored when thinking is enabled.
        max_tokens: Optional max_tokens override for the response portion.
    """
    is_openrouter = model_id.startswith("openrouter:")
    is_anthropic = "anthropic" in model_id and not is_openrouter
    is_google = model_id.startswith("google-gla:") or model_id.startswith("google-vertex:")

    # Compute effective temperature and max_tokens
    base_max_tokens = max_tokens if max_tokens is not None else _RESPONSE_MAX_TOKENS
    if thinking_budget:
        effective_max_tokens = thinking_budget + base_max_tokens
        # Temperature 1.0 is required by Anthropic extended thinking;
        # for non-Anthropic models, use the default temperature
        effective_temperature = _THINKING_TEMPERATURE if is_anthropic else _DEFAULT_TEMPERATURE
    else:
        effective_temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
        effective_max_tokens = base_max_tokens

    if is_anthropic:
        anthropic_kwargs: dict[str, Any] = {
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            "anthropic_cache_instructions": True,
            "anthropic_cache_tool_definitions": True,
            # Explicit timeout bypasses SDK client-side heuristic that rejects
            # non-streaming requests when max_tokens exceeds ~21k.
            "timeout": httpx.Timeout(1200.0, connect=5.0),
        }

        # Only include thinking when enabled — passing None triggers a 400 error.
        if thinking_budget:
            anthropic_kwargs["anthropic_thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        settings: (
            AnthropicModelSettings | GoogleModelSettings | OpenRouterModelSettings | dict[str, Any]
        ) = AnthropicModelSettings(**anthropic_kwargs)

        logger.info(
            "LLM model settings built",
            provider="anthropic",
            thinking="enabled" if thinking_budget else "disabled",
            thinking_budget=thinking_budget,
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )

    elif is_openrouter:
        openrouter_kwargs: dict[str, Any] = {
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
            "openrouter_provider": {
                "data_collection": "deny",
                "require_parameters": True,
            },
        }
        if thinking_budget:
            openrouter_kwargs["openrouter_reasoning"] = {"effort": "high"}

        settings = OpenRouterModelSettings(**openrouter_kwargs)

        logger.info(
            "LLM model settings built",
            provider="openrouter",
            reasoning="enabled" if thinking_budget else "disabled",
            temperature=effective_temperature,
            max_tokens=effective_max_tokens,
        )

    elif is_google:
        google_kwargs: dict[str, Any] = {
            "temperature": effective_temperature,
            "max_tokens": base_max_tokens,
        }
        if thinking_budget:
            google_kwargs["google_thinking_config"] = {
                "include_thoughts": True,
                "thinking_budget": thinking_budget,
            }

        settings = GoogleModelSettings(**google_kwargs)

        logger.info(
            "LLM model settings built",
            provider="google",
            thinking="enabled" if thinking_budget else "disabled",
            thinking_budget=thinking_budget,
            temperature=effective_temperature,
            max_tokens=base_max_tokens,
        )

    else:
        if thinking_budget:
            logger.warning(
                "Thinking budget configured for unsupported provider — will be ignored",
                model=model_id,
                thinking_budget=thinking_budget,
            )
        # Generic providers don't support thinking — use base max_tokens
        settings = {
            "temperature": effective_temperature,
            "max_tokens": base_max_tokens,
        }

    return settings
