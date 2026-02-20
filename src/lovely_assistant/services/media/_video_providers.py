"""Video generation provider implementations.

Each provider has a submit function (start generation) and a poll function
(check status). Providers are standalone async functions — same pattern as
image providers in _providers.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from loguru import logger

from lovely_assistant.services.media.exceptions import (
    ProviderError,
    ProviderNotConfiguredError,
    VideoJobError,
)


@dataclass
class VideoStatus:
    """Snapshot of a video generation job's current state."""

    state: str  # "submitted", "processing", "completed", "failed"
    video_url: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Runway
# ---------------------------------------------------------------------------


async def submit_runway(prompt: str, model_name: str, duration: int) -> str:
    """Submit a video generation job to Runway.

    Returns the provider's task ID.
    """
    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        raise ProviderNotConfiguredError("RUNWAYML_API_SECRET environment variable is not set")

    try:
        from runwayml import AsyncRunwayML

        client = AsyncRunwayML(api_key=api_key)
        task = await client.image_to_video.create(
            model=model_name,
            prompt_text=prompt,
            duration=duration,
        )
        job_id = task.id
        logger.info("Runway video job submitted", job_id=job_id, model=model_name)
        return job_id

    except ImportError as e:
        raise ProviderError(f"Runway SDK not available: {e}") from e
    except Exception as e:
        if isinstance(e, (ProviderError, ProviderNotConfiguredError)):
            raise
        raise ProviderError(f"Runway API error: {e}") from e


async def poll_runway(job_id: str) -> VideoStatus:
    """Poll Runway for the status of a video generation task."""
    api_key = os.environ.get("RUNWAYML_API_SECRET")
    if not api_key:
        raise ProviderNotConfiguredError("RUNWAYML_API_SECRET environment variable is not set")

    try:
        from runwayml import AsyncRunwayML

        client = AsyncRunwayML(api_key=api_key)
        task = await client.tasks.retrieve(id=job_id)

        status = task.status.upper() if task.status else "UNKNOWN"

        if status == "SUCCEEDED":
            video_url = task.output[0] if task.output else None
            if not video_url:
                raise VideoJobError(f"Runway task {job_id} succeeded but returned no output URL")
            return VideoStatus(state="completed", video_url=video_url)

        if status == "FAILED":
            return VideoStatus(state="failed", error=f"Runway task {job_id} failed")

        # PENDING, RUNNING, or anything else → processing
        return VideoStatus(state="processing")

    except ImportError as e:
        raise ProviderError(f"Runway SDK not available: {e}") from e
    except Exception as e:
        if isinstance(e, (ProviderError, ProviderNotConfiguredError, VideoJobError)):
            raise
        raise ProviderError(f"Runway poll error: {e}") from e


# ---------------------------------------------------------------------------
# Luma
# ---------------------------------------------------------------------------


async def submit_luma(prompt: str, model_name: str, duration: int) -> str:
    """Submit a video generation job to Luma AI.

    Returns the provider's generation ID.
    """
    api_key = os.environ.get("LUMAAI_API_KEY")
    if not api_key:
        raise ProviderNotConfiguredError("LUMAAI_API_KEY environment variable is not set")

    try:
        from lumaai import AsyncLumaAI

        client = AsyncLumaAI(auth_token=api_key)
        generation = await client.generations.create(
            model=model_name,
            prompt=prompt,
        )
        job_id = generation.id
        logger.info("Luma video job submitted", job_id=job_id, model=model_name)
        return job_id

    except ImportError as e:
        raise ProviderError(f"Luma SDK not available: {e}") from e
    except Exception as e:
        if isinstance(e, (ProviderError, ProviderNotConfiguredError)):
            raise
        raise ProviderError(f"Luma API error: {e}") from e


async def poll_luma(job_id: str) -> VideoStatus:
    """Poll Luma AI for the status of a video generation."""
    api_key = os.environ.get("LUMAAI_API_KEY")
    if not api_key:
        raise ProviderNotConfiguredError("LUMAAI_API_KEY environment variable is not set")

    try:
        from lumaai import AsyncLumaAI

        client = AsyncLumaAI(auth_token=api_key)
        generation = await client.generations.get(id=job_id)

        state = generation.state.lower() if generation.state else "unknown"

        if state == "completed":
            video_url = None
            if generation.assets and hasattr(generation.assets, "video"):
                video_url = generation.assets.video
            if not video_url:
                raise VideoJobError(f"Luma generation {job_id} completed but returned no video URL")
            return VideoStatus(state="completed", video_url=video_url)

        if state == "failed":
            reason = getattr(generation, "failure_reason", None) or "Unknown failure"
            return VideoStatus(state="failed", error=reason)

        # queued, dreaming, or anything else → processing
        return VideoStatus(state="processing")

    except ImportError as e:
        raise ProviderError(f"Luma SDK not available: {e}") from e
    except Exception as e:
        if isinstance(e, (ProviderError, ProviderNotConfiguredError, VideoJobError)):
            raise
        raise ProviderError(f"Luma poll error: {e}") from e
