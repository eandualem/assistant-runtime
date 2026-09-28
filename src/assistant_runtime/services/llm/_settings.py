"""Internal model settings construction for Pydantic AI agents.

Translated from an earlier assistant implementation's agent_factory.py (build_model_settings)
and agent_config.py (validation, constants).

Not part of the public module API — imported only by interface.py.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx
from loguru import logger
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.models.google import GoogleModelSettings
from pydantic_ai.models.openrouter import OpenRouterModelSettings
from pydantic_ai.profiles.anthropic import anthropic_model_profile

from assistant_runtime.services.llm.exceptions import ProviderConfigError

# The output limit when the application sets no max_tokens (Anthropic requires one).
_RESPONSE_MAX_TOKENS = 8_192


def _map_openai_reasoning_effort(*, model_id: str, thinking_budget: int) -> str:
    """Map the assistant's numeric thinking budget onto OpenAI reasoning effort tiers."""
    if model_id.endswith("-pro"):
        # OpenAI's Pro reasoning models operate at fixed high effort.
        return "high"
    if thinking_budget <= 4_000:
        return "low"
    if thinking_budget <= 12_000:
        return "medium"
    if thinking_budget <= 32_000:
        return "high"
    return "xhigh"


def _map_anthropic_effort(*, thinking_budget: int, supports_xhigh: bool) -> str:
    """Map the numeric thinking budget onto Anthropic effort levels for adaptive thinking."""
    if thinking_budget <= 4_000:
        return "low"
    if thinking_budget <= 12_000:
        return "medium"
    if thinking_budget <= 32_000 or not supports_xhigh:
        return "high"
    return "xhigh"


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

    # pydantic-ai 2.x uses google: (Gemini API) and google-cloud: (Vertex); the 1.x
    # prefixes google-gla: / google-vertex: are no longer recognised.
    for legacy, current in (("google-gla:", "google:"), ("google-vertex:", "google-cloud:")):
        if model_id.startswith(legacy):
            raise ProviderConfigError(
                f"Use '{current}' prefix, not '{legacy}': '{model_id}'. "
                f"Use '{current}{model_id[len(legacy) :]}'"
            )

    # Validate colon-separated format
    if ":" not in model_id:
        raise ProviderConfigError(
            f"Invalid model ID: '{model_id}'. "
            f"Must use 'provider:model-name' format (e.g., 'anthropic:claude-sonnet-5')."
        )
    parts = model_id.split(":", 1)
    if not parts[0] or not parts[1]:
        raise ProviderConfigError(
            f"Invalid model ID: '{model_id}'. "
            f"Must use 'provider:model-name' format (e.g., 'anthropic:claude-sonnet-5')."
        )

    return model_id


def build_model_settings(
    *,
    model_id: str,
    thinking_budget: int | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    base: Mapping[str, Any] | None = None,
) -> AnthropicModelSettings | GoogleModelSettings | OpenRouterModelSettings | dict[str, Any]:
    """Build provider-specific model settings for a Pydantic AI Agent.

    Three layers, the later one winning: the runtime's defaults for the
    provider (output limit, caching, transport options), the application's
    ``base`` (native Pydantic AI ``ModelSettings``), then the tunables. A
    tunable is sent only when set; nothing is substituted for it.

    Args:
        model_id: Full model identifier (e.g., "anthropic:claude-sonnet-4-6").
        thinking_budget: Optional thinking token budget. None sends no thinking setting,
            so the provider's default applies.
        temperature: Optional temperature. None sends none. Left out where the model
            rejects it (Anthropic with thinking, models without sampling settings).
        max_tokens: Optional max_tokens for the response portion; overrides ``base``.
        base: The application's model settings, passed to Pydantic AI unchanged.
    """
    is_openrouter = model_id.startswith("openrouter:")
    is_anthropic = "anthropic" in model_id and not is_openrouter
    is_google = model_id.startswith("google:") or model_id.startswith("google-cloud:")
    is_astra = model_id == "openai:gpt-6-astra" or model_id.startswith("openai:gpt-6-astra-")
    is_openai_reasoning = model_id.startswith("openai:gpt-5") or is_astra

    base = dict(base or {})
    # The response portion; a thinking budget that shares the output limit is added to it.
    response_tokens = (
        max_tokens if max_tokens is not None else base.get("max_tokens", _RESPONSE_MAX_TOKENS)
    )

    if is_anthropic:
        model_name = model_id.split(":", 1)[1]
        profile = dict(anthropic_model_profile(model_name) or {})
        adaptive = bool(profile.get("anthropic_supports_adaptive_thinking", False))
        no_sampling = bool(profile.get("anthropic_disallows_sampling_settings", False))
        supports_xhigh = bool(profile.get("anthropic_supports_xhigh_effort", False))

        anthropic_kwargs: dict[str, Any] = {
            "anthropic_cache_instructions": True,
            "anthropic_cache_tool_definitions": True,
            # A cache point on the conversation, moved forward as it grows.
            "anthropic_cache": True,
            # Explicit timeout bypasses SDK client-side heuristic that rejects
            # non-streaming requests when max_tokens exceeds ~21k.
            "timeout": httpx.Timeout(1200.0, connect=5.0),
            **base,
            "max_tokens": response_tokens,
        }
        # Opus 4.7+, Sonnet 5 and the Fable/Mythos family reject temperature/top_p/top_k,
        # and extended thinking requires the default temperature.
        if temperature is not None and not no_sampling and not thinking_budget:
            anthropic_kwargs["temperature"] = temperature

        # Only include thinking when enabled — passing None triggers a 400 error.
        if thinking_budget:
            if adaptive:
                # 4.6+ models: adaptive thinking; the numeric budget becomes an effort level.
                anthropic_kwargs["anthropic_thinking"] = {"type": "adaptive"}
                anthropic_kwargs["anthropic_effort"] = _map_anthropic_effort(
                    thinking_budget=thinking_budget, supports_xhigh=supports_xhigh
                )
            else:
                # budget_tokens counts against max_tokens, so the response keeps its share.
                anthropic_kwargs["anthropic_thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
                anthropic_kwargs["max_tokens"] = thinking_budget + response_tokens

        settings: (
            AnthropicModelSettings | GoogleModelSettings | OpenRouterModelSettings | dict[str, Any]
        ) = AnthropicModelSettings(**anthropic_kwargs)

        logger.info(
            "LLM model settings built",
            provider="anthropic",
            thinking=(
                ("adaptive" if adaptive else "enabled")
                if thinking_budget
                # Opus 5 / Sonnet 5 / Fable run adaptive thinking when the field is omitted.
                else ("provider_default" if adaptive else "disabled")
            ),
            thinking_budget=thinking_budget,
            effort=anthropic_kwargs.get("anthropic_effort"),
            temperature=anthropic_kwargs.get("temperature"),
            max_tokens=anthropic_kwargs["max_tokens"],
        )

    elif is_openrouter:
        openrouter_kwargs: dict[str, Any] = {
            "openrouter_provider": {
                "data_collection": "deny",
                "require_parameters": True,
            },
            **base,
            "max_tokens": response_tokens,
        }
        if temperature is not None:
            openrouter_kwargs["temperature"] = temperature
        if thinking_budget:
            openrouter_kwargs["openrouter_reasoning"] = {"max_tokens": thinking_budget}
            openrouter_kwargs["max_tokens"] = thinking_budget + response_tokens

        settings = OpenRouterModelSettings(**openrouter_kwargs)

        logger.info(
            "LLM model settings built",
            provider="openrouter",
            reasoning="enabled" if thinking_budget else "disabled",
            temperature=openrouter_kwargs.get("temperature"),
            max_tokens=openrouter_kwargs["max_tokens"],
        )

    elif is_google:
        google_kwargs: dict[str, Any] = {**base, "max_tokens": response_tokens}
        if temperature is not None:
            google_kwargs["temperature"] = temperature
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
            temperature=google_kwargs.get("temperature"),
            max_tokens=response_tokens,
        )

    elif is_openai_reasoning:
        openai_kwargs: dict[str, Any] = {
            # Let OpenAI resume from the most recent response state so tool
            # continuations don't resend the entire prior conversation.
            "openai_previous_response_id": "auto",
            # We compact and sanitize stored history, so replaying provider item
            # IDs can break OpenAI continuation requests.
            "openai_send_reasoning_ids": False,
            **base,
            "max_tokens": response_tokens,
        }
        if temperature is not None and not is_astra:
            openai_kwargs["temperature"] = temperature
        if thinking_budget:
            openai_kwargs["openai_reasoning_effort"] = _map_openai_reasoning_effort(
                model_id=model_id,
                thinking_budget=thinking_budget,
            )
            openai_kwargs["openai_reasoning_summary"] = "detailed"

        settings = openai_kwargs

        logger.info(
            "LLM model settings built",
            provider="openai",
            reasoning="enabled" if thinking_budget else "disabled",
            thinking_budget=thinking_budget,
            reasoning_effort=openai_kwargs.get("openai_reasoning_effort"),
            max_tokens=response_tokens,
        )

    else:
        if thinking_budget:
            logger.warning(
                "Thinking budget configured for unsupported provider — will be ignored",
                model=model_id,
                thinking_budget=thinking_budget,
            )
        settings = {**base, "max_tokens": response_tokens}
        if temperature is not None:
            settings["temperature"] = temperature

    return settings
