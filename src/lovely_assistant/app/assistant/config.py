"""Configuration for the assistant module."""

from pydantic import BaseModel, ConfigDict, Field


class AssistantConfig(BaseModel):
    """Assistant module configuration. Nested into AppSettings as `assistant`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_model: str | None = Field(
        default=None,
        description="Model override for assistant conversations. None = use LLM primary_model.",
    )
    thinking_budget: int | None = Field(
        default=10000,
        ge=1,
        le=100000,
        description="Thinking token budget for extended thinking. None = disabled.",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        description="LLM temperature for response generation.",
    )
    max_turns: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum agent loop turns per request.",
    )
    enable_working_memory: bool = Field(
        default=True,
        description="Whether to extract working memory deltas after each turn.",
    )
    session_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Session TTL in hours.",
    )
    pending_tool_call_timeout_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Minutes before a pending frontend tool call expires.",
    )
