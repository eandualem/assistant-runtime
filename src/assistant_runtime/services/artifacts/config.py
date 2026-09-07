"""Configuration for the artifact service module."""

from pydantic import BaseModel, ConfigDict, Field


class ArtifactsConfig(BaseModel):
    """Artifact module configuration. Nested into AppSettings as `artifacts`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cache_ttl_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=3600.0,
        description=(
            "How long active texts read from the store are reused by prompts. "
            "Mutations through the service invalidate the cache immediately."
        ),
    )
    history_limit: int = Field(default=20, ge=1, le=100)
