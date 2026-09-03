"""Tests for streaming exception hierarchy."""

from assistant_runtime.app.streaming.exceptions import (
    EventLimitError,
    StreamExecutionError,
    StreamingError,
    StreamSetupError,
)
from assistant_runtime.base.exceptions import AssistantRuntimeError


class TestStreamingError:
    def test_is_assistant_runtime_error(self):
        assert issubclass(StreamingError, AssistantRuntimeError)

    def test_default_category(self):
        err = StreamingError("test")
        assert err.category == "streaming"

    def test_default_severity(self):
        err = StreamingError("test")
        assert err.severity == "high"


class TestStreamSetupError:
    def test_is_streaming_error(self):
        assert issubclass(StreamSetupError, StreamingError)

    def test_severity_medium(self):
        err = StreamSetupError("setup fail")
        assert err.severity == "medium"

    def test_category_inherited(self):
        err = StreamSetupError("setup fail")
        assert err.category == "streaming"


class TestStreamExecutionError:
    def test_is_streaming_error(self):
        assert issubclass(StreamExecutionError, StreamingError)

    def test_default_severity_high(self):
        err = StreamExecutionError("exec fail")
        assert err.severity == "high"


class TestEventLimitError:
    def test_is_streaming_error(self):
        assert issubclass(EventLimitError, StreamingError)

    def test_severity_medium(self):
        err = EventLimitError("too many events")
        assert err.severity == "medium"

    def test_retry_not_allowed(self):
        err = EventLimitError("too many events")
        assert err.retry_allowed is False
