"""Configuration for the LLM service module."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from assistant_runtime.model_catalog import ALLOWED_PROVIDERS


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(..., description="Provider name (anthropic, openai, google, openrouter)")
    api_key: SecretStr = Field(..., description="Provider API key")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        if v not in ALLOWED_PROVIDERS:
            raise ValueError(f"Provider must be one of {ALLOWED_PROVIDERS}, got '{v}'")
        return v


class LLMConfig(BaseModel):
    """LLM module configuration. Nested into AppSettings as `llm`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary_model: str = Field(
        default="anthropic:claude-opus-5",
        description="Default model for assistant conversations",
    )
    summarization_model: str = Field(
        default="anthropic:claude-haiku-4-5",
        description="Model for history summarization and lightweight tasks",
    )
    providers_json: str | None = Field(
        default=None,
        description="JSON string with provider configurations (alternative to individual env vars)",
    )
    codex_models: list[str] = Field(
        default_factory=list,
        description=(
            "OpenAI model names routed through the ChatGPT/Codex subscription when it is "
            "connected. Empty (the default) routes every openai: model that way; the backend "
            "decides what the subscription allows."
        ),
    )
    codex_only: bool = Field(
        default=False,
        description=(
            "The subscription guard: every openai: model must go through the ChatGPT/Codex "
            "subscription, never an OPENAI_API_KEY, and a disconnected subscription is an "
            "error rather than an API fallback. Other providers with a configured key stay "
            "routable; ASSISTANT__REQUEST_MODELS limits what a request may pick. "
            "Does not govern separate voice or media services."
        ),
    )

    codex_service_tier: Literal["default", "fast"] | None = Field(
        default=None,
        description=(
            "Requested Codex subscription service tier. Unset omits the wire field; "
            "fast requests priority processing with higher subscription credit consumption. "
            "Only affects Codex-authenticated LLM calls, not API-key or voice/media calls."
        ),
    )
