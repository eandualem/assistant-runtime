"""Tests for StreamingConfig validation, defaults, and bounds."""

import pytest
from pydantic import ValidationError

from lovely_assistant.app.streaming.config import StreamingConfig


class TestStreamingConfigDefaults:
    def test_defaults(self):
        config = StreamingConfig()
        assert config.debounce_seconds == 0.05
        assert config.max_events_per_stream == 10000
        assert config.stream_timeout_seconds == 300.0

    def test_custom_values(self):
        config = StreamingConfig(
            debounce_seconds=0.1,
            max_events_per_stream=5000,
            stream_timeout_seconds=60.0,
        )
        assert config.debounce_seconds == 0.1
        assert config.max_events_per_stream == 5000
        assert config.stream_timeout_seconds == 60.0


class TestStreamingConfigValidation:
    def test_frozen(self):
        config = StreamingConfig()
        with pytest.raises(ValidationError):
            config.debounce_seconds = 0.1

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StreamingConfig(nonexistent_field="value")

    def test_debounce_below_min_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(debounce_seconds=-0.1)

    def test_debounce_above_max_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(debounce_seconds=1.5)

    def test_debounce_boundary_min(self):
        config = StreamingConfig(debounce_seconds=0.0)
        assert config.debounce_seconds == 0.0

    def test_debounce_boundary_max(self):
        config = StreamingConfig(debounce_seconds=1.0)
        assert config.debounce_seconds == 1.0

    def test_max_events_below_min_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(max_events_per_stream=50)

    def test_max_events_above_max_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(max_events_per_stream=200000)

    def test_max_events_boundary_min(self):
        config = StreamingConfig(max_events_per_stream=100)
        assert config.max_events_per_stream == 100

    def test_max_events_boundary_max(self):
        config = StreamingConfig(max_events_per_stream=100000)
        assert config.max_events_per_stream == 100000

    def test_timeout_below_min_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(stream_timeout_seconds=5.0)

    def test_timeout_above_max_rejected(self):
        with pytest.raises(ValidationError):
            StreamingConfig(stream_timeout_seconds=700.0)

    def test_timeout_boundary_min(self):
        config = StreamingConfig(stream_timeout_seconds=10.0)
        assert config.stream_timeout_seconds == 10.0

    def test_timeout_boundary_max(self):
        config = StreamingConfig(stream_timeout_seconds=600.0)
        assert config.stream_timeout_seconds == 600.0
