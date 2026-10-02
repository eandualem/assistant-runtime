"""Tests for structured logging helpers — tag constants and payload truncation."""

from assistant_runtime.base._logging import FULL_LOG_THRESHOLD, truncate


class TestTruncate:
    def test_short_text_unchanged(self):
        text = "hello"
        assert truncate(text) == "hello"

    def test_long_text_truncated(self):
        text = "x" * (FULL_LOG_THRESHOLD + 100)
        result = truncate(text)
        assert result.startswith("x" * FULL_LOG_THRESHOLD)
        assert "100 chars truncated" in result
        assert len(result) < len(text)

    def test_exact_length_unchanged(self):
        text = "a" * FULL_LOG_THRESHOLD
        assert truncate(text) == text
