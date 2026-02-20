"""Tests for MediaService video generation methods: generate_video, get_video_status, _parse_video_model_id."""

from unittest.mock import AsyncMock, patch

import pytest

from lovely_assistant.services.media._video_providers import VideoStatus
from lovely_assistant.services.media.config import MediaConfig
from lovely_assistant.services.media.exceptions import MediaError, ProviderError
from lovely_assistant.services.media.interface import MediaService
from lovely_assistant.services.media.models import VideoResult, VideoStatusResponse

MODULE = "lovely_assistant.services.media.interface"


@pytest.fixture
def config():
    return MediaConfig()


@pytest.fixture
def service(config):
    return MediaService(config=config)


@pytest.fixture
async def started_service(service):
    await service.start()
    yield service
    await service.stop()


# ---------------------------------------------------------------------------
# Video Lifecycle
# ---------------------------------------------------------------------------


class TestVideoLifecycle:
    async def test_start_initializes_job_tracker(self, service):
        await service.start()
        assert service._job_tracker is not None
        await service.stop()

    async def test_stop_clears_job_tracker(self, service):
        await service.start()
        await service.stop()
        assert service._job_tracker is None

    async def test_stop_calls_stop_all_on_tracker(self, service):
        await service.start()
        tracker = service._job_tracker
        tracker.stop_all = AsyncMock()
        await service.stop()
        tracker.stop_all.assert_called_once()

    async def test_health_check_includes_active_video_jobs(self, started_service):
        health = await started_service.health_check()
        assert "active_video_jobs" in health
        assert health["active_video_jobs"] == 0


# ---------------------------------------------------------------------------
# generate_video
# ---------------------------------------------------------------------------


class TestGenerateVideo:
    async def test_not_started_raises_media_error(self, service):
        with pytest.raises(MediaError, match="not started"):
            await service.generate_video(prompt="a sunset")

    async def test_uses_default_video_model_from_config(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="job-123",
            ) as mock_submit,
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            await started_service.generate_video(prompt="a sunset")
            mock_submit.assert_called_once()
            call_kwargs = mock_submit.call_args.kwargs
            assert call_kwargs["model_name"] == "gen4-turbo"

    async def test_overrides_model(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_luma",
                new_callable=AsyncMock,
                return_value="luma-job-456",
            ) as mock_submit,
            patch(
                f"{MODULE}.poll_luma",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            await started_service.generate_video(prompt="a sunset", model="luma:ray-2")
            mock_submit.assert_called_once()
            call_kwargs = mock_submit.call_args.kwargs
            assert call_kwargs["model_name"] == "ray-2"

    async def test_routes_to_runway_provider(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="job-123",
            ) as mock_submit,
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            await started_service.generate_video(prompt="a sunset", model="runway:gen4-turbo")
            mock_submit.assert_called_once_with(
                prompt="a sunset", model_name="gen4-turbo", duration=5
            )

    async def test_routes_to_luma_provider(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_luma",
                new_callable=AsyncMock,
                return_value="luma-job-456",
            ) as mock_submit,
            patch(
                f"{MODULE}.poll_luma",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            await started_service.generate_video(prompt="ocean waves", model="luma:ray-2")
            mock_submit.assert_called_once_with(
                prompt="ocean waves", model_name="ray-2", duration=5
            )

    async def test_returns_video_result_with_correct_fields(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="job-123",
            ),
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            result = await started_service.generate_video(
                prompt="a sunset", model="runway:gen4-turbo"
            )

        assert isinstance(result, VideoResult)
        assert result.job_id  # non-empty
        assert result.status == "submitted"
        assert result.provider == "runway"
        assert result.model == "runway:gen4-turbo"

    async def test_status_url_format(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="job-123",
            ),
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            result = await started_service.generate_video(
                prompt="a sunset", model="runway:gen4-turbo"
            )

        assert result.status_url.startswith("/api/media/video/")
        assert result.status_url == f"/api/media/video/{result.job_id}"

    async def test_passes_duration_to_provider(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="job-123",
            ) as mock_submit,
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            await started_service.generate_video(
                prompt="a sunset", model="runway:gen4-turbo", duration=10
            )
            call_kwargs = mock_submit.call_args.kwargs
            assert call_kwargs["duration"] == 10


# ---------------------------------------------------------------------------
# _parse_video_model_id
# ---------------------------------------------------------------------------


class TestParseVideoModelId:
    def test_runway_model(self):
        provider, model_name = MediaService._parse_video_model_id("runway:gen4-turbo")
        assert provider == "runway"
        assert model_name == "gen4-turbo"

    def test_luma_model(self):
        provider, model_name = MediaService._parse_video_model_id("luma:ray-2")
        assert provider == "luma"
        assert model_name == "ray-2"

    def test_no_colon_raises_provider_error(self):
        with pytest.raises(ProviderError, match="Invalid model ID"):
            MediaService._parse_video_model_id("gen4-turbo")

    def test_unknown_video_provider_raises_provider_error(self):
        with pytest.raises(ProviderError, match="Unknown video provider"):
            MediaService._parse_video_model_id("midjourney:v6")

    def test_colon_in_model_name_preserves_full_name(self):
        provider, model_name = MediaService._parse_video_model_id("runway:model:variant")
        assert provider == "runway"
        assert model_name == "model:variant"


# ---------------------------------------------------------------------------
# get_video_status
# ---------------------------------------------------------------------------


class TestGetVideoStatus:
    def test_returns_none_when_not_started(self, service):
        result = service.get_video_status("some-job-id")
        assert result is None

    async def test_returns_none_for_missing_job_id(self, started_service):
        result = started_service.get_video_status("nonexistent-job-id")
        assert result is None

    async def test_returns_status_for_existing_job(self, started_service):
        with (
            patch(
                f"{MODULE}.submit_runway",
                new_callable=AsyncMock,
                return_value="provider-job-abc",
            ),
            patch(
                f"{MODULE}.poll_runway",
                new_callable=AsyncMock,
                return_value=VideoStatus(state="processing"),
            ),
        ):
            video_result = await started_service.generate_video(
                prompt="a sunset", model="runway:gen4-turbo"
            )

        status = started_service.get_video_status(video_result.job_id)
        assert status is not None
        assert isinstance(status, VideoStatusResponse)
        assert status.job_id == video_result.job_id
        assert status.provider == "runway"
        assert status.model == "runway:gen4-turbo"
