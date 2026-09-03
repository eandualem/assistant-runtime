"""Configuration for the assistant heartbeat module."""

from pydantic import BaseModel, ConfigDict, Field


class HeartbeatConfig(BaseModel):
    """Configuration for periodic assistant heartbeat injection."""

    enabled: bool = Field(default=True, description="Whether the heartbeat loop is enabled.")
    interval_seconds: int = Field(
        default=300,
        ge=30,
        description="Seconds between heartbeat injections.",
    )
    startup_delay_seconds: int = Field(
        default=300,
        ge=0,
        description="Seconds to wait before the first heartbeat after startup.",
    )
    message: str = Field(
        default="[via:heartbeat]",
        min_length=1,
        description="Injected heartbeat message body.",
    )
    from_agent: str = Field(
        default="heartbeat",
        min_length=1,
        description="Synthetic sender used for heartbeat inbox items.",
    )

    model_config = ConfigDict(frozen=True, extra="forbid")
