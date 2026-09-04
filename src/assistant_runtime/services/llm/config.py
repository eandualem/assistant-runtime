"""Configuration for the LLM service module."""

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

ALLOWED_PROVIDERS = ["anthropic", "openai", "google", "openrouter"]

# Maps provider names to the environment variables Pydantic AI reads for auto-detection.
# When keys are loaded from LLM_PROVIDERS_JSON, they must be exported to these env vars
# so that Pydantic AI's provider constructors can find them.
# The model used when the configured primary/summarization model's provider has
# no credentials but another provider does. Order = preference.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "anthropic:claude-opus-5",
    "openai": "openai:gpt-5.6-terra",
    "google": "google:gemini-3.1-pro-preview",
    "openrouter": "openrouter:x-ai/grok-4.1-fast",
}
PROVIDER_DEFAULT_SUMMARIZATION_MODELS: dict[str, str] = {
    "anthropic": "anthropic:claude-haiku-4-5",
    "openai": "openai:gpt-5.6-luna",
    "google": "google:gemini-3.8-flash",
    "openrouter": "openrouter:x-ai/grok-4.1-fast",
}

_PROVIDER_ENV_VAR_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(..., description="Provider name (anthropic, openai, google, openrouter)")
    api_key: SecretStr = Field(..., description="Provider API key")
    base_url: str | None = Field(default=None, description="Custom base URL for the provider API")
    timeout: float = Field(default=120.0, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")

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
