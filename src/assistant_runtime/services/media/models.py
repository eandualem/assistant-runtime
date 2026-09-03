"""Data models for the media service module."""

from pydantic import BaseModel


class GeneratedImage(BaseModel):
    """Raw image data returned from a provider."""

    image_bytes: bytes
    mime_type: str  # "image/png", "image/jpeg", "image/webp"
    provider: str  # "openai", "google"
    model: str  # full model ID e.g. "openai:gpt-image-1"


class MediaResult(BaseModel):
    """Result returned to callers after generation + caching."""

    image_id: str
    url: str  # relative: "/api/media/{image_id}"
    provider: str
    model: str
    mime_type: str


class VideoResult(BaseModel):
    """Result returned when a video generation job is submitted."""

    job_id: str  # internal UUID for status polling
    status: str  # "submitted"
    provider: str
    model: str
    status_url: str  # "/api/media/video/{job_id}"


class VideoStatusResponse(BaseModel):
    """Response from the video status endpoint."""

    job_id: str
    state: str  # "submitted", "processing", "completed", "failed"
    video_url: str | None = None
    error: str | None = None
    provider: str
    model: str
