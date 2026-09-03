"""Static model registry — single source of truth for available AI models.

Provides model catalog with capability metadata, provider availability detection,
and current defaults. No lifecycle management — pure data + env var reads.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from assistant_runtime.config import AppSettings


class ModelEntry(BaseModel):
    """A single model in the registry."""

    model_config = ConfigDict(frozen=True)

    id: str
    provider: str
    name: str
    capability: str  # tier: flagship | balanced | fast
    context_window: int | None = None
    capabilities: list[
        str
    ]  # feature tags: text, vision, thinking, image-generation, video-generation
    description: str


class ProviderInfo(BaseModel):
    """Provider availability information."""

    model_config = ConfigDict(frozen=True)

    name: str
    env_var: str
    configured: bool


# Maps provider name → env var name. Broader than llm/config.py's map
# (includes image/video providers that aren't Pydantic AI providers).
_REGISTRY_PROVIDER_ENV_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "runway": "RUNWAYML_API_SECRET",
    "luma": "LUMAAI_API_KEY",
}

_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "openrouter": "OpenRouter",
    "runway": "Runway",
    "luma": "Luma",
}

MODEL_CATALOG: list[ModelEntry] = [
    # --- Anthropic ---
    ModelEntry(
        id="anthropic:claude-opus-4-6",
        provider="anthropic",
        name="Claude Opus 4.6",
        capability="flagship",
        context_window=200_000,
        capabilities=["text", "vision", "thinking"],
        description="Most capable Claude model — deep reasoning and analysis",
    ),
    ModelEntry(
        id="anthropic:claude-sonnet-4-6",
        provider="anthropic",
        name="Claude Sonnet 4.6",
        capability="balanced",
        context_window=200_000,
        capabilities=["text", "vision", "thinking"],
        description="Balanced performance and speed with extended thinking",
    ),
    ModelEntry(
        id="anthropic:claude-sonnet-4-5",
        provider="anthropic",
        name="Claude Sonnet 4.5",
        capability="balanced",
        context_window=200_000,
        capabilities=["text", "vision", "thinking"],
        description="Previous-generation balanced model",
    ),
    ModelEntry(
        id="anthropic:claude-haiku-4-5",
        provider="anthropic",
        name="Claude Haiku 4.5",
        capability="fast",
        context_window=200_000,
        capabilities=["text", "vision"],
        description="Fastest Claude model — low latency for operational tasks",
    ),
    # --- OpenAI (GPT-5.4 only) ---
    ModelEntry(
        id="openai:gpt-5.4",
        provider="openai",
        name="GPT-5.4",
        capability="flagship",
        context_window=1_050_000,
        capabilities=["text", "vision", "thinking"],
        description="OpenAI's flagship — 1M context, reasoning effort: none/low/medium/high/xhigh",
    ),
    ModelEntry(
        id="openai:gpt-5.4-pro",
        provider="openai",
        name="GPT-5.4 Pro",
        capability="flagship",
        context_window=1_050_000,
        capabilities=["text", "vision", "thinking"],
        description="Maximum compute variant — best on hard reasoning, async recommended",
    ),
    # --- Google text ---
    ModelEntry(
        id="google:gemini-3-pro-preview",
        provider="google",
        name="Gemini 3 Pro Preview",
        capability="flagship",
        context_window=1_000_000,
        capabilities=["text", "vision", "thinking"],
        description="Google's flagship model with 1M token context",
    ),
    ModelEntry(
        id="google:gemini-3.1-pro-preview",
        provider="google",
        name="Gemini 3.1 Pro Preview",
        capability="flagship",
        context_window=1_000_000,
        capabilities=["text", "vision", "thinking"],
        description="Google's latest flagship — enhanced reasoning and agentic workflows",
    ),
    ModelEntry(
        id="google:gemini-3-flash-preview",
        provider="google",
        name="Gemini 3 Flash Preview",
        capability="balanced",
        context_window=1_000_000,
        capabilities=["text", "vision", "thinking"],
        description="Fast Gemini variant with 1M token context",
    ),
    # --- OpenRouter ---
    ModelEntry(
        id="openrouter:deepseek/deepseek-v3.2",
        provider="openrouter",
        name="DeepSeek V3.2",
        capability="balanced",
        context_window=128_000,
        capabilities=["text"],
        description="DeepSeek's latest open model via OpenRouter",
    ),
    ModelEntry(
        id="openrouter:mimo-ai/mimo-v2-flash",
        provider="openrouter",
        name="Mimo V2 Flash",
        capability="balanced",
        context_window=128_000,
        capabilities=["text", "vision"],
        description="Multimodal model optimized for speed",
    ),
    ModelEntry(
        id="openrouter:x-ai/grok-4",
        provider="openrouter",
        name="Grok 4",
        capability="flagship",
        context_window=256_000,
        capabilities=["text", "vision", "thinking"],
        description="xAI's flagship reasoning model",
    ),
    ModelEntry(
        id="openrouter:x-ai/grok-4-fast",
        provider="openrouter",
        name="Grok 4 Fast",
        capability="balanced",
        context_window=256_000,
        capabilities=["text", "vision"],
        description="Speed-optimized Grok 4 variant",
    ),
    ModelEntry(
        id="openrouter:x-ai/grok-4.1-fast",
        provider="openrouter",
        name="Grok 4.1 Fast",
        capability="balanced",
        context_window=256_000,
        capabilities=["text", "vision"],
        description="Latest fast Grok iteration",
    ),
    ModelEntry(
        id="openrouter:zhipu/glm-4.7",
        provider="openrouter",
        name="GLM 4.7",
        capability="balanced",
        context_window=128_000,
        capabilities=["text"],
        description="Zhipu's general-purpose language model",
    ),
    ModelEntry(
        id="openrouter:moonshotai/kimi-k2.5",
        provider="openrouter",
        name="Kimi K2.5",
        capability="balanced",
        context_window=128_000,
        capabilities=["text", "thinking"],
        description="Moonshot's reasoning-capable model",
    ),
    ModelEntry(
        id="openrouter:nvidia/nemotron-3-nano",
        provider="openrouter",
        name="Nemotron 3 Nano",
        capability="fast",
        context_window=128_000,
        capabilities=["text"],
        description="NVIDIA's compact efficient model",
    ),
    ModelEntry(
        id="openrouter:qwen/qwen3-coder",
        provider="openrouter",
        name="Qwen3 Coder",
        capability="balanced",
        context_window=128_000,
        capabilities=["text"],
        description="Code-specialized Qwen variant",
    ),
    # --- OpenAI image generation ---
    ModelEntry(
        id="openai:gpt-image-1.5",
        provider="openai",
        name="GPT Image 1.5",
        capability="flagship",
        capabilities=["image-generation"],
        description="Latest OpenAI image generation model",
    ),
    ModelEntry(
        id="openai:gpt-image-1",
        provider="openai",
        name="GPT Image 1",
        capability="balanced",
        capabilities=["image-generation"],
        description="Standard OpenAI image generation",
    ),
    ModelEntry(
        id="openai:gpt-image-1-mini",
        provider="openai",
        name="GPT Image 1 Mini",
        capability="fast",
        capabilities=["image-generation"],
        description="Fast, cost-effective image generation",
    ),
    # --- Google image generation ---
    ModelEntry(
        id="google:imagen-4.0-generate-001",
        provider="google",
        name="Imagen 4.0",
        capability="balanced",
        capabilities=["image-generation"],
        description="Google's standard image generation model",
    ),
    ModelEntry(
        id="google:imagen-4.0-ultra-generate-001",
        provider="google",
        name="Imagen 4.0 Ultra",
        capability="flagship",
        capabilities=["image-generation"],
        description="Highest quality Google image generation",
    ),
    ModelEntry(
        id="google:imagen-4.0-fast-generate-001",
        provider="google",
        name="Imagen 4.0 Fast",
        capability="fast",
        capabilities=["image-generation"],
        description="Speed-optimized Google image generation",
    ),
    # --- Runway video generation ---
    ModelEntry(
        id="runway:gen4.5",
        provider="runway",
        name="Runway Gen-4.5",
        capability="flagship",
        capabilities=["video-generation"],
        description="Runway's latest video generation model",
    ),
    ModelEntry(
        id="runway:gen4-turbo",
        provider="runway",
        name="Runway Gen-4 Turbo",
        capability="balanced",
        capabilities=["video-generation"],
        description="Speed-optimized Runway video generation",
    ),
    # --- Luma video generation ---
    ModelEntry(
        id="luma:ray-2",
        provider="luma",
        name="Luma Ray 2",
        capability="flagship",
        capabilities=["video-generation"],
        description="Luma's flagship video generation model",
    ),
    ModelEntry(
        id="luma:ray-flash-2",
        provider="luma",
        name="Luma Ray Flash 2",
        capability="fast",
        capabilities=["video-generation"],
        description="Fast Luma video generation",
    ),
]


def get_models(
    capability: str | None = None,
    provider: str | None = None,
) -> list[ModelEntry]:
    """Filter MODEL_CATALOG by capability tag and/or provider.

    Args:
        capability: Filter by capabilities list membership (contains check).
        provider: Filter by provider field (exact match).

    Returns:
        Filtered list of ModelEntry instances. Empty list if no match.
    """
    results = MODEL_CATALOG
    if capability is not None:
        results = [m for m in results if capability in m.capabilities]
    if provider is not None:
        results = [m for m in results if m.provider == provider]
    return results


def get_provider_info() -> dict[str, ProviderInfo]:
    """Check provider availability by testing env var presence.

    Returns:
        Dict mapping provider name → ProviderInfo with configured status.
    """
    return {
        provider: ProviderInfo(
            name=_PROVIDER_DISPLAY_NAMES[provider],
            env_var=env_var,
            configured=bool(os.getenv(env_var)),
        )
        for provider, env_var in _REGISTRY_PROVIDER_ENV_MAP.items()
    }


def get_defaults() -> dict[str, Any]:
    """Return current default model settings from AppSettings.

    Constructs AppSettings at request time to reflect live env var state.
    """
    settings = AppSettings()
    return {
        "primary_model": settings.llm.primary_model,
        "summarization_model": settings.llm.summarization_model,
        "working_memory_model": settings.history.working_memory_model,
        "default_image_model": settings.media.default_image_model,
        "default_video_model": settings.media.default_video_model,
        "subagent_model": None,
        "subagent_thinking_budget": None,
    }
