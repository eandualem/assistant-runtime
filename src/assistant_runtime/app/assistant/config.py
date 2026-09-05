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
    profile: str | None = Field(
        default=None,
        description=(
            "Assistant profile: a built-in name (neutral, technical_operator) or the path "
            "of a TOML profile file. None = neutral. An AssistantDefinition.profile wins."
        ),
    )
    session_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Session TTL in hours.",
    )


class TunableOverrides(BaseModel):
    """The settings a client may override, per request or at runtime.

    This is the one definition of the tunables: the request body's ``config``,
    the ``PATCH /api/settings`` body, the runtime overlay's validation and the
    resolved ``EffectiveConfig`` all derive from it. ``None`` means "not set".
    """

    model_config = ConfigDict(extra="forbid")

    default_model: str | None = Field(default=None, description="Model id (provider:name)")
    thinking_budget: int | None = Field(default=None, ge=1, le=100_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_turns: int | None = Field(default=None, ge=1, le=50)
    enable_working_memory: bool | None = None
    summarization_model: str | None = None
    working_memory_model: str | None = None
    default_image_model: str | None = None
    default_video_model: str | None = None
    subagent_model: str | None = None
    subagent_thinking_budget: int | None = Field(default=None, ge=1, le=100_000)


TUNABLE_FIELDS: frozenset[str] = frozenset(TunableOverrides.model_fields)
