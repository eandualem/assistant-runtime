"""Configuration for the tasks module."""

from pydantic import BaseModel, ConfigDict, Field


class TasksConfig(BaseModel):
    """Background tasks: turns that run outside any conversation turn. Nested as ``tasks``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "Offer the task and agent tools and accept POST /api/tasks and /api/agents. "
            "Each task or agent message runs model turns."
        ),
    )
    max_concurrent: int = Field(
        default=4, ge=1, le=32, description="Tasks running at once; the rest wait in order."
    )
    max_waiting: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Queued and running tasks beyond which starts fail.",
    )
    timeout_seconds: int = Field(
        default=900, ge=10, le=86_400, description="A task's own time limit (not the tool timeout)."
    )
    result_max_chars: int = Field(
        default=20_000, ge=100, le=1_000_000, description="The stored result is cut to this length."
    )
