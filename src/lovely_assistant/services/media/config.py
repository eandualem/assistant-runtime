"""Configuration for the media service module."""

from pydantic import BaseModel, ConfigDict, Field


class MediaConfig(BaseModel):
    """Media module configuration. Nested into AppSettings as `media`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    default_image_model: str = Field(default="openai:gpt-image-1")
    default_size: str = Field(default="1024x1024")
    default_quality: str = Field(default="medium")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    cache_max_items: int = Field(default=100, ge=1, le=1000)
    generation_timeout_seconds: float = Field(default=60.0, ge=5.0, le=300.0)

    # Video generation settings
    default_video_model: str = Field(default="runway:gen4-turbo")
    video_poll_interval_seconds: float = Field(default=5.0, ge=2.0, le=30.0)
    video_timeout_seconds: float = Field(default=300.0, ge=30.0, le=600.0)
    video_max_concurrent_jobs: int = Field(default=5, ge=1, le=20)
