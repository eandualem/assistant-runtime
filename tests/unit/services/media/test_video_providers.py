"""Tests for video generation provider functions (Runway and Luma)."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lovely_assistant.services.media._video_providers import (
    VideoStatus,
    poll_luma,
    poll_runway,
    submit_luma,
    submit_runway,
)
from lovely_assistant.services.media.exceptions import (
    ProviderError,
    ProviderNotConfiguredError,
    VideoJobError,
)

MODULE = "lovely_assistant.services.media._video_providers"


class TestSubmitRunway:
    """Tests for the submit_runway provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="RUNWAYML_API_SECRET"):
            await submit_runway(
                prompt="a cinematic landscape",
                model_name="gen4_turbo",
                duration=10,
            )

    async def test_successful_submission(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(id="task-abc-123")

        mock_client = MagicMock()
        mock_client.image_to_video.create = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            result = await submit_runway(
                prompt="a cinematic landscape",
                model_name="gen4_turbo",
                duration=10,
            )

        assert result == "task-abc-123"

    async def test_api_key_passed_to_client(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-my-specific-key")

        mock_task = SimpleNamespace(id="task-xyz")

        mock_client = MagicMock()
        mock_client.image_to_video.create = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client) as mock_cls:
            await submit_runway(
                prompt="test",
                model_name="gen4_turbo",
                duration=5,
            )

            mock_cls.assert_called_once_with(api_key="rw-my-specific-key")

    async def test_create_call_parameters(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(id="task-params")

        mock_client = MagicMock()
        mock_client.image_to_video.create = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            await submit_runway(
                prompt="a beautiful sunset over the ocean",
                model_name="gen4_turbo",
                duration=10,
            )

            mock_client.image_to_video.create.assert_called_once_with(
                model="gen4_turbo",
                prompt_text="a beautiful sunset over the ocean",
                duration=10,
            )

    async def test_generic_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_client = MagicMock()
        mock_client.image_to_video.create = AsyncMock(
            side_effect=Exception("Connection timeout"),
        )

        with (
            patch("runwayml.AsyncRunwayML", return_value=mock_client),
            pytest.raises(ProviderError, match="Runway API error"),
        ):
            await submit_runway(
                prompt="test",
                model_name="gen4_turbo",
                duration=5,
            )

    async def test_import_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        with (
            patch.dict(sys.modules, {"runwayml": None}),
            pytest.raises(ProviderError, match="Runway SDK not available"),
        ):
            await submit_runway(
                prompt="test",
                model_name="gen4_turbo",
                duration=5,
            )


class TestPollRunway:
    """Tests for the poll_runway provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("RUNWAYML_API_SECRET", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="RUNWAYML_API_SECRET"):
            await poll_runway(job_id="task-abc-123")

    async def test_succeeded_returns_completed_with_url(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(
            status="SUCCEEDED",
            output=["https://cdn.runway.com/video-abc.mp4"],
        )

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            result = await poll_runway(job_id="task-abc-123")

        assert isinstance(result, VideoStatus)
        assert result.state == "completed"
        assert result.video_url == "https://cdn.runway.com/video-abc.mp4"

    async def test_failed_returns_failed_state(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(status="FAILED", output=None)

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            result = await poll_runway(job_id="task-fail-1")

        assert result.state == "failed"
        assert result.error is not None
        assert "task-fail-1" in result.error

    async def test_pending_returns_processing(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(status="PENDING", output=None)

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            result = await poll_runway(job_id="task-pending")

        assert result.state == "processing"

    async def test_running_returns_processing(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(status="RUNNING", output=None)

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(return_value=mock_task)

        with patch("runwayml.AsyncRunwayML", return_value=mock_client):
            result = await poll_runway(job_id="task-running")

        assert result.state == "processing"

    async def test_succeeded_no_output_raises_video_job_error(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_task = SimpleNamespace(status="SUCCEEDED", output=[])

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(return_value=mock_task)

        with (
            patch("runwayml.AsyncRunwayML", return_value=mock_client),
            pytest.raises(VideoJobError, match="no output URL"),
        ):
            await poll_runway(job_id="task-no-output")

    async def test_generic_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("RUNWAYML_API_SECRET", "rw-test-key")

        mock_client = MagicMock()
        mock_client.tasks.retrieve = AsyncMock(
            side_effect=Exception("Network failure"),
        )

        with (
            patch("runwayml.AsyncRunwayML", return_value=mock_client),
            pytest.raises(ProviderError, match="Runway poll error"),
        ):
            await poll_runway(job_id="task-error")


class TestSubmitLuma:
    """Tests for the submit_luma provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("LUMAAI_API_KEY", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="LUMAAI_API_KEY"):
            await submit_luma(
                prompt="a flowing river",
                model_name="ray-2",
                duration=5,
            )

    async def test_successful_submission(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_generation = SimpleNamespace(id="gen-luma-456")

        mock_client = MagicMock()
        mock_client.generations.create = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            result = await submit_luma(
                prompt="a flowing river",
                model_name="ray-2",
                duration=5,
            )

        assert result == "gen-luma-456"

    async def test_api_key_passed_to_client(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-my-specific-key")

        mock_generation = SimpleNamespace(id="gen-xyz")

        mock_client = MagicMock()
        mock_client.generations.create = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client) as mock_cls:
            await submit_luma(
                prompt="test",
                model_name="ray-2",
                duration=5,
            )

            mock_cls.assert_called_once_with(auth_token="luma-my-specific-key")

    async def test_create_call_parameters(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_generation = SimpleNamespace(id="gen-params")

        mock_client = MagicMock()
        mock_client.generations.create = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            await submit_luma(
                prompt="a majestic eagle soaring",
                model_name="ray-2",
                duration=5,
            )

            mock_client.generations.create.assert_called_once_with(
                model="ray-2",
                prompt="a majestic eagle soaring",
            )

    async def test_generic_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_client = MagicMock()
        mock_client.generations.create = AsyncMock(
            side_effect=Exception("Connection refused"),
        )

        with (
            patch("lumaai.AsyncLumaAI", return_value=mock_client),
            pytest.raises(ProviderError, match="Luma API error"),
        ):
            await submit_luma(
                prompt="test",
                model_name="ray-2",
                duration=5,
            )

    async def test_import_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        with (
            patch.dict(sys.modules, {"lumaai": None}),
            pytest.raises(ProviderError, match="Luma SDK not available"),
        ):
            await submit_luma(
                prompt="test",
                model_name="ray-2",
                duration=5,
            )


class TestPollLuma:
    """Tests for the poll_luma provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("LUMAAI_API_KEY", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="LUMAAI_API_KEY"):
            await poll_luma(job_id="gen-luma-456")

    async def test_completed_returns_completed_with_url(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_assets = SimpleNamespace(video="https://cdn.lumaai.com/video-456.mp4")
        mock_generation = SimpleNamespace(
            state="completed",
            assets=mock_assets,
            failure_reason=None,
        )

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            result = await poll_luma(job_id="gen-luma-456")

        assert isinstance(result, VideoStatus)
        assert result.state == "completed"
        assert result.video_url == "https://cdn.lumaai.com/video-456.mp4"

    async def test_failed_returns_failed_with_reason(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_generation = SimpleNamespace(
            state="failed",
            assets=None,
            failure_reason="Content policy violation detected",
        )

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            result = await poll_luma(job_id="gen-fail-1")

        assert result.state == "failed"
        assert result.error == "Content policy violation detected"

    async def test_queued_returns_processing(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_generation = SimpleNamespace(
            state="queued",
            assets=None,
            failure_reason=None,
        )

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            result = await poll_luma(job_id="gen-queued")

        assert result.state == "processing"

    async def test_dreaming_returns_processing(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_generation = SimpleNamespace(
            state="dreaming",
            assets=None,
            failure_reason=None,
        )

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(return_value=mock_generation)

        with patch("lumaai.AsyncLumaAI", return_value=mock_client):
            result = await poll_luma(job_id="gen-dreaming")

        assert result.state == "processing"

    async def test_completed_no_video_url_raises_video_job_error(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_assets = SimpleNamespace(video=None)
        mock_generation = SimpleNamespace(
            state="completed",
            assets=mock_assets,
            failure_reason=None,
        )

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(return_value=mock_generation)

        with (
            patch("lumaai.AsyncLumaAI", return_value=mock_client),
            pytest.raises(VideoJobError, match="no video URL"),
        ):
            await poll_luma(job_id="gen-no-url")

    async def test_generic_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("LUMAAI_API_KEY", "luma-test-key")

        mock_client = MagicMock()
        mock_client.generations.get = AsyncMock(
            side_effect=Exception("Service unavailable"),
        )

        with (
            patch("lumaai.AsyncLumaAI", return_value=mock_client),
            pytest.raises(ProviderError, match="Luma poll error"),
        ):
            await poll_luma(job_id="gen-error")
