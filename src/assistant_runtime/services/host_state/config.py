"""Configuration for the host state module."""

from pydantic import BaseModel, ConfigDict, Field


class HostStateConfig(BaseModel):
    """Values a host keeps in the runtime. Nested into AppSettings as ``host_state``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_value_bytes: int = Field(
        default=262_144, ge=1024, le=16_777_216, description="Largest value, as UTF-8 JSON."
    )
    max_page: int = Field(
        default=1000, ge=1, le=10_000, description="Most keys one listing returns."
    )
