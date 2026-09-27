"""Configuration for the event log module."""

from pydantic import BaseModel, ConfigDict, Field


class EventLogConfig(BaseModel):
    """The event log a host writes. Nested into AppSettings as ``event_log``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_page: int = Field(
        default=500, ge=1, le=5000, description="Most events one listing returns."
    )
