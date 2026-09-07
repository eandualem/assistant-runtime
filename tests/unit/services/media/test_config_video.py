"""Tests for video-related media service configuration fields."""

import pytest
from pydantic import ValidationError

from assistant_runtime.services.media.config import MediaConfig


class TestVideoConfigDefaults:
    def test_default_video_model(self):
        config = MediaConfig()
        assert config.default_video_model == "runway:gen4-turbo"

    def test_default_video_poll_interval(self):
        config = MediaConfig()
        assert config.video_poll_interval_seconds == 5.0

    def test_default_video_timeout(self):
        config = MediaConfig()
        assert config.video_timeout_seconds == 300.0

    def test_default_video_max_concurrent_jobs(self):
        config = MediaConfig()
        assert config.video_max_concurrent_jobs == 5

    def test_override_video_model(self):
        config = MediaConfig(default_video_model="luma:ray-2")
        assert config.default_video_model == "luma:ray-2"

    def test_override_video_poll_interval(self):
        config = MediaConfig(video_poll_interval_seconds=10.0)
        assert config.video_poll_interval_seconds == 10.0

    def test_override_video_timeout(self):
        config = MediaConfig(video_timeout_seconds=120.0)
        assert config.video_timeout_seconds == 120.0

    def test_override_video_max_concurrent_jobs(self):
        config = MediaConfig(video_max_concurrent_jobs=10)
        assert config.video_max_concurrent_jobs == 10


class TestVideoConfigValidation:
    # --- video_poll_interval_seconds boundaries (ge=2.0, le=30.0) ---

    def test_poll_interval_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_poll_interval_seconds=1.9)

    def test_poll_interval_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_poll_interval_seconds=30.1)

    def test_poll_interval_boundary_min(self):
        config = MediaConfig(video_poll_interval_seconds=2.0)
        assert config.video_poll_interval_seconds == 2.0

    def test_poll_interval_boundary_max(self):
        config = MediaConfig(video_poll_interval_seconds=30.0)
        assert config.video_poll_interval_seconds == 30.0

    # --- video_timeout_seconds boundaries (ge=30.0, le=600.0) ---

    def test_timeout_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_timeout_seconds=29.9)

    def test_timeout_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_timeout_seconds=600.1)

    def test_timeout_boundary_min(self):
        config = MediaConfig(video_timeout_seconds=30.0)
        assert config.video_timeout_seconds == 30.0

    def test_timeout_boundary_max(self):
        config = MediaConfig(video_timeout_seconds=600.0)
        assert config.video_timeout_seconds == 600.0

    # --- video_max_concurrent_jobs boundaries (ge=1, le=20) ---

    def test_max_concurrent_below_min_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_max_concurrent_jobs=0)

    def test_max_concurrent_above_max_rejected(self):
        with pytest.raises(ValidationError):
            MediaConfig(video_max_concurrent_jobs=21)

    def test_max_concurrent_boundary_min(self):
        config = MediaConfig(video_max_concurrent_jobs=1)
        assert config.video_max_concurrent_jobs == 1

    def test_max_concurrent_boundary_max(self):
        config = MediaConfig(video_max_concurrent_jobs=20)
        assert config.video_max_concurrent_jobs == 20
