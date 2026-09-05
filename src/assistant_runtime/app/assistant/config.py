"""Configuration for the assistant module."""

from pydantic import BaseModel, ConfigDict, Field


class UsageBudget(BaseModel):
    """Per-turn ceilings the host sets, in native Pydantic AI terms.

    Every field is optional; unset means no limit of that kind. ``max_turns``
    (the request limit) stays a tunable on ``AssistantConfig``. Cost limits
    only apply when the provider reports a price for the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_calls: int | None = Field(default=None, ge=0, description="Tool calls per turn")
    input_tokens: int | None = Field(default=None, ge=1, description="Input tokens per turn")
    output_tokens: int | None = Field(default=None, ge=1, description="Output tokens per turn")
    total_tokens: int | None = Field(default=None, ge=1, description="Total tokens per turn")
    cost_usd: float | None = Field(default=None, gt=0, description="Reported cost per turn, USD")


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
    budget: UsageBudget = Field(
        default_factory=UsageBudget,
        description="Per-turn usage ceilings (ASSISTANT__BUDGET__*); see docs/concepts.md.",
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
