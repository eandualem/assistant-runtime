"""Configuration for the LLM service module."""

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from assistant_runtime.model_catalog import ALLOWED_PROVIDERS


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
