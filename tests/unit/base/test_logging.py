"""Tests for structured logging helpers — tag constants and payload truncation."""

from assistant_runtime.base._logging import (
    FULL_LOG_THRESHOLD,
    TAG_DB,
    TAG_HISTORY,
    TAG_LLM,
    TAG_REQUEST,
    TAG_SESSION,
    TAG_STREAM,
    TAG_TOOLS,
    truncate,
    truncate_payload,
)


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


class TestTruncatePayload:
    def test_converts_non_string_to_string(self):
        result = truncate_payload(12345)
        assert result == "12345"

    def test_truncates_long_payload(self):
        payload = {"key": "v" * 1000}
        result = truncate_payload(payload)
        assert "chars truncated" in result


class TestTagConstants:
    def test_all_tags_bracketed(self):
        tags = [TAG_REQUEST, TAG_STREAM, TAG_TOOLS, TAG_SESSION, TAG_DB, TAG_LLM, TAG_HISTORY]
        for tag in tags:
            assert tag.startswith("["), f"{tag} does not start with '['"
            assert tag.endswith("]"), f"{tag} does not end with ']'"
