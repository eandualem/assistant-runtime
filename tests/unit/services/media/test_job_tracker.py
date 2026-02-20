"""Tests for JobTracker background video job management."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from lovely_assistant.services.media._job_tracker import JobTracker, VideoJob
from lovely_assistant.services.media._video_providers import VideoStatus
from lovely_assistant.services.media.exceptions import VideoJobError


@pytest.fixture
def tracker():
    return JobTracker(max_concurrent=5, poll_interval=0.01, timeout=10.0)


@pytest.fixture
def mock_poll_fn():
    return AsyncMock(return_value=VideoStatus(state="processing"))


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class TestJobTrackerSubmit:
    async def test_submit_creates_job_with_internal_id(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            assert isinstance(job, VideoJob)
            assert job.internal_id  # non-empty UUID hex
            assert len(job.internal_id) == 32  # uuid4().hex is 32 chars
        finally:
            await tracker.stop_all()

    async def test_submit_starts_background_task(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            assert job.poll_task is not None
            assert not job.poll_task.done()
        finally:
            await tracker.stop_all()

    async def test_submit_returns_submitted_state(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            assert job.state == "submitted"
        finally:
            await tracker.stop_all()

    async def test_submit_stores_provider_and_model(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4-turbo", poll_fn=mock_poll_fn
        )
        try:
            assert job.provider == "runway"
            assert job.model == "runway:gen4-turbo"
            assert job.provider_job_id == "prov-123"
        finally:
            await tracker.stop_all()

    async def test_max_concurrent_jobs_raises_error(self, mock_poll_fn):
        small_tracker = JobTracker(max_concurrent=2, poll_interval=0.01, timeout=10.0)
        try:
            small_tracker.submit("j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)
            small_tracker.submit("j2", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)

            with pytest.raises(VideoJobError, match="Maximum concurrent"):
                small_tracker.submit(
                    "j3", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
                )
        finally:
            await small_tracker.stop_all()

    async def test_completed_jobs_dont_count_against_limit(self, mock_poll_fn):
        small_tracker = JobTracker(max_concurrent=2, poll_interval=0.01, timeout=10.0)
        try:
            job1 = small_tracker.submit(
                "j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
            )
            # Manually mark job1 as completed so it no longer counts as active
            job1.state = "completed"

            small_tracker.submit("j2", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)
            # This should succeed because job1 is completed (only 1 active)
            job3 = small_tracker.submit(
                "j3", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
            )
            assert job3.state == "submitted"
        finally:
            await small_tracker.stop_all()


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestJobTrackerGetStatus:
    async def test_get_existing_job(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            result = tracker.get_status(job.internal_id)
            assert result is job
        finally:
            await tracker.stop_all()

    async def test_get_nonexistent_returns_none(self, tracker):
        result = tracker.get_status("nonexistent-id")
        assert result is None


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


class TestJobTrackerCancel:
    async def test_cancel_existing_job_returns_true(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            result = tracker.cancel(job.internal_id)
            assert result is True
        finally:
            await tracker.stop_all()

    async def test_cancel_nonexistent_returns_false(self, tracker):
        result = tracker.cancel("nonexistent-id")
        assert result is False

    async def test_cancel_sets_failed_state(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            tracker.cancel(job.internal_id)
            assert job.state == "failed"
        finally:
            await tracker.stop_all()

    async def test_cancel_sets_cancelled_error(self, tracker, mock_poll_fn):
        job = tracker.submit(
            "prov-123", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
        )
        try:
            tracker.cancel(job.internal_id)
            assert job.error == "Cancelled"
        finally:
            await tracker.stop_all()


# ---------------------------------------------------------------------------
# stop_all
# ---------------------------------------------------------------------------


class TestJobTrackerStopAll:
    async def test_stop_all_cancels_running_tasks(self, tracker, mock_poll_fn):
        job1 = tracker.submit("j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)
        job2 = tracker.submit("j2", provider="luma", model="luma:ray2", poll_fn=mock_poll_fn)

        await tracker.stop_all()

        assert job1.poll_task.done()
        assert job2.poll_task.done()

    async def test_stop_all_with_no_jobs(self, tracker):
        # Should not raise any errors
        await tracker.stop_all()


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestJobTrackerLen:
    async def test_len_empty(self, tracker):
        assert len(tracker) == 0

    async def test_len_after_submit(self, tracker, mock_poll_fn):
        try:
            tracker.submit("j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)
            tracker.submit("j2", provider="luma", model="luma:ray2", poll_fn=mock_poll_fn)
            assert len(tracker) == 2
        finally:
            await tracker.stop_all()


# ---------------------------------------------------------------------------
# active_count
# ---------------------------------------------------------------------------


class TestJobTrackerActiveCount:
    async def test_active_count_with_submitted_jobs(self, tracker, mock_poll_fn):
        try:
            tracker.submit("j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn)
            tracker.submit("j2", provider="luma", model="luma:ray2", poll_fn=mock_poll_fn)
            assert tracker.active_count() == 2
        finally:
            await tracker.stop_all()

    async def test_active_count_excludes_completed(self, tracker, mock_poll_fn):
        try:
            job1 = tracker.submit(
                "j1", provider="runway", model="runway:gen4", poll_fn=mock_poll_fn
            )
            tracker.submit("j2", provider="luma", model="luma:ray2", poll_fn=mock_poll_fn)
            # Manually mark job1 as completed
            job1.state = "completed"
            assert tracker.active_count() == 1
        finally:
            await tracker.stop_all()


# ---------------------------------------------------------------------------
# _poll_loop
# ---------------------------------------------------------------------------


class TestPollLoop:
    async def test_poll_loop_completes_job(self):
        tracker = JobTracker(max_concurrent=5, poll_interval=0.01, timeout=10.0)
        call_count = 0

        async def poll_fn(job_id: str) -> VideoStatus:
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return VideoStatus(state="completed", video_url="https://example.com/video.mp4")
            return VideoStatus(state="processing")

        job = tracker.submit("prov-1", provider="runway", model="runway:gen4", poll_fn=poll_fn)
        try:
            # Wait for the poll loop to complete the job
            await asyncio.wait_for(job.poll_task, timeout=2.0)
            assert job.state == "completed"
            assert job.video_url == "https://example.com/video.mp4"
        finally:
            await tracker.stop_all()

    async def test_poll_loop_fails_job(self):
        tracker = JobTracker(max_concurrent=5, poll_interval=0.01, timeout=10.0)

        async def poll_fn(job_id: str) -> VideoStatus:
            return VideoStatus(state="failed", error="Generation failed on provider")

        job = tracker.submit("prov-1", provider="runway", model="runway:gen4", poll_fn=poll_fn)
        try:
            await asyncio.wait_for(job.poll_task, timeout=2.0)
            assert job.state == "failed"
            assert job.error == "Generation failed on provider"
        finally:
            await tracker.stop_all()

    async def test_poll_loop_timeout(self):
        tracker = JobTracker(max_concurrent=5, poll_interval=0.01, timeout=0.05)

        async def poll_fn(job_id: str) -> VideoStatus:
            return VideoStatus(state="processing")

        job = tracker.submit("prov-1", provider="runway", model="runway:gen4", poll_fn=poll_fn)
        try:
            await asyncio.wait_for(job.poll_task, timeout=2.0)
            assert job.state == "failed"
            assert "timed out" in job.error.lower()
        finally:
            await tracker.stop_all()

    async def test_poll_loop_handles_poll_error(self):
        tracker = JobTracker(max_concurrent=5, poll_interval=0.01, timeout=10.0)

        async def poll_fn(job_id: str) -> VideoStatus:
            raise RuntimeError("Provider API unreachable")

        job = tracker.submit("prov-1", provider="runway", model="runway:gen4", poll_fn=poll_fn)
        try:
            await asyncio.wait_for(job.poll_task, timeout=2.0)
            assert job.state == "failed"
            assert "Poll error" in job.error
            assert "Provider API unreachable" in job.error
        finally:
            await tracker.stop_all()

    async def test_poll_loop_transitions_from_submitted_to_processing(self):
        tracker = JobTracker(max_concurrent=5, poll_interval=0.01, timeout=10.0)
        call_count = 0

        async def poll_fn(job_id: str) -> VideoStatus:
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                return VideoStatus(state="completed", video_url="https://example.com/v.mp4")
            return VideoStatus(state="processing")

        job = tracker.submit("prov-1", provider="runway", model="runway:gen4", poll_fn=poll_fn)
        try:
            # Let at least one poll iteration run
            await asyncio.sleep(0.05)
            # After processing response, state should transition from submitted to processing
            # (unless it already completed)
            assert job.state in ("processing", "completed")

            # Wait for completion
            await asyncio.wait_for(job.poll_task, timeout=2.0)
            assert job.state == "completed"
        finally:
            await tracker.stop_all()
