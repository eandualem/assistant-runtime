"""Background job tracker for async video generation.

Manages in-flight video generation jobs. Each job gets an asyncio.Task
that polls the provider until terminal state (completed/failed) or timeout.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from assistant_runtime.services.media._video_providers import VideoStatus
from assistant_runtime.services.media.exceptions import VideoJobError, VideoTimeoutError


@dataclass
class VideoJob:
    """Tracks a single video generation job."""

    internal_id: str  # our UUID for status endpoint
    provider_job_id: str  # provider's job ID
    provider: str
    model: str
    state: str = "submitted"  # submitted, processing, completed, failed
    video_url: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    poll_task: asyncio.Task | None = field(default=None, repr=False)


class JobTracker:
    """Manages background polling tasks for video generation jobs."""

    def __init__(
        self,
        max_concurrent: int,
        poll_interval: float,
        timeout: float,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._poll_interval = poll_interval
        self._timeout = timeout
        self._jobs: dict[str, VideoJob] = {}

    def submit(
        self,
        provider_job_id: str,
        provider: str,
        model: str,
        poll_fn: Callable[..., Coroutine[Any, Any, VideoStatus]],
    ) -> VideoJob:
        """Register a new video job and start background polling.

        Args:
            provider_job_id: The provider's job/task ID.
            provider: Provider name (e.g. "runway", "luma").
            model: Full model ID (e.g. "runway:gen4-turbo").
            poll_fn: Async callable that takes (job_id) and returns VideoStatus.

        Returns:
            The VideoJob with an internal_id and background poll task started.

        Raises:
            VideoJobError: If max concurrent jobs exceeded.
        """
        active_count = sum(1 for j in self._jobs.values() if j.state in ("submitted", "processing"))
        if active_count >= self._max_concurrent:
            raise VideoJobError(f"Maximum concurrent video jobs ({self._max_concurrent}) reached")

        internal_id = uuid.uuid4().hex
        job = VideoJob(
            internal_id=internal_id,
            provider_job_id=provider_job_id,
            provider=provider,
            model=model,
        )
        self._jobs[internal_id] = job

        # Spawn background polling task
        job.poll_task = asyncio.create_task(
            self._poll_loop(job, poll_fn),
            name=f"video-poll-{internal_id[:8]}",
        )

        logger.info(
            "Video job submitted",
            internal_id=internal_id,
            provider_job_id=provider_job_id,
            provider=provider,
        )
        return job

    def get_status(self, internal_id: str) -> VideoJob | None:
        """Get the current state of a video job by internal ID."""
        return self._jobs.get(internal_id)

    def cancel(self, internal_id: str) -> bool:
        """Cancel a video job's polling task.

        Returns True if the job was found and cancelled, False if not found.
        """
        job = self._jobs.get(internal_id)
        if job is None:
            return False
        if job.poll_task and not job.poll_task.done():
            job.poll_task.cancel()
        if job.state in ("submitted", "processing"):
            job.state = "failed"
            job.error = "Cancelled"
            job.updated_at = time.monotonic()
        return True

    async def stop_all(self) -> None:
        """Cancel all background polling tasks. Called during shutdown."""
        tasks_to_cancel = []
        for job in self._jobs.values():
            if job.poll_task and not job.poll_task.done():
                job.poll_task.cancel()
                tasks_to_cancel.append(job.poll_task)

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            logger.info("Cancelled video polling tasks", count=len(tasks_to_cancel))

    def __len__(self) -> int:
        """Total number of tracked jobs (all states)."""
        return len(self._jobs)

    def active_count(self) -> int:
        """Number of jobs still in non-terminal state."""
        return sum(1 for j in self._jobs.values() if j.state in ("submitted", "processing"))

    async def _poll_loop(
        self,
        job: VideoJob,
        poll_fn: Callable[..., Coroutine[Any, Any, VideoStatus]],
    ) -> None:
        """Background loop: poll provider until terminal state or timeout."""
        deadline = job.created_at + self._timeout

        try:
            while True:
                await asyncio.sleep(self._poll_interval)

                # Check timeout
                if time.monotonic() > deadline:
                    job.state = "failed"
                    job.error = "Video generation timed out"
                    job.updated_at = time.monotonic()
                    logger.warning(
                        "Video job timed out",
                        internal_id=job.internal_id,
                        timeout=self._timeout,
                    )
                    raise VideoTimeoutError(
                        f"Video job {job.internal_id} timed out after {self._timeout}s"
                    )

                try:
                    status = await poll_fn(job.provider_job_id)
                except Exception as e:
                    logger.error(
                        "Video poll error",
                        internal_id=job.internal_id,
                        error=str(e),
                    )
                    job.state = "failed"
                    job.error = f"Poll error: {e}"
                    job.updated_at = time.monotonic()
                    return

                job.updated_at = time.monotonic()

                if status.state == "completed":
                    job.state = "completed"
                    job.video_url = status.video_url
                    logger.info(
                        "Video job completed",
                        internal_id=job.internal_id,
                        video_url=status.video_url,
                    )
                    return

                if status.state == "failed":
                    job.state = "failed"
                    job.error = status.error or "Provider reported failure"
                    logger.warning(
                        "Video job failed",
                        internal_id=job.internal_id,
                        error=job.error,
                    )
                    return

                # Still processing
                if job.state == "submitted":
                    job.state = "processing"

        except asyncio.CancelledError:
            logger.debug("Video poll task cancelled", internal_id=job.internal_id)
            raise
        except VideoTimeoutError:
            return  # already updated job state above
