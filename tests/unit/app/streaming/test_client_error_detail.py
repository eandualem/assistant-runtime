"""``STREAMING__CLIENT_ERROR_DETAIL``: what error text a client is allowed to see."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from assistant_runtime.app.streaming._runner import _describe_error
from assistant_runtime.app.streaming.config import StreamingConfig
from assistant_runtime.app.streaming.interface import StreamingService
from assistant_runtime.services.llm.exceptions import LLMCallError


class TestDescribeError:
    def test_detail_on_keeps_the_exception_text(self):
        message, error_type, retry = _describe_error(RuntimeError("boom /private/path"))
        assert message == "Request failed: RuntimeError: boom /private/path"
        assert error_type == "internal"
        assert retry is False

    def test_detail_off_hides_internal_text(self):
        message, error_type, _ = _describe_error(RuntimeError("boom /private/path"), detail=False)
        assert message == "Request failed"
        assert error_type == "internal"

    def test_detail_off_keeps_the_provider_category_only(self):
        exc = LLMCallError(
            "LLM call failed (RATE_LIMIT): org-1234 quota",
            error_category="RATE_LIMIT",
            is_retryable=True,
        )
        message, error_type, retry = _describe_error(exc, detail=False)
        assert message == "LLM call failed (RATE_LIMIT)"
        assert error_type == "rate_limit"
        assert retry is True
        assert "org-1234" in _describe_error(exc)[0]


class TestTraceRetention:
    @pytest.mark.asyncio
    async def test_without_a_reachable_database_nothing_runs(self):
        service = StreamingService(
            StreamingConfig(), MagicMock(), MagicMock(), MagicMock(), database_service=None
        )
        assert await service.cleanup_expired_traces() == 0
        unreachable = MagicMock(healthy=False)
        service = StreamingService(
            StreamingConfig(), MagicMock(), MagicMock(), MagicMock(), database_service=unreachable
        )
        assert await service.cleanup_expired_traces() == 0
        unreachable.session_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletes_rows_older_than_the_retention(self, monkeypatch):
        from contextlib import asynccontextmanager

        from assistant_runtime.services.database import repositories

        seen: dict[str, datetime] = {}

        class FakeTraces:
            def __init__(self, session) -> None:
                pass

            async def delete_older_than(self, cutoff: datetime) -> int:
                seen["cutoff"] = cutoff
                return 3

        monkeypatch.setattr(repositories, "TraceRepository", FakeTraces)

        @asynccontextmanager
        async def session_context():
            yield object()

        db = MagicMock(healthy=True, session_context=session_context)
        service = StreamingService(
            StreamingConfig(trace_retention_hours=48),
            MagicMock(),
            MagicMock(),
            MagicMock(),
            database_service=db,
        )
        before = datetime.now(UTC)
        assert await service.cleanup_expired_traces() == 3
        assert timedelta(hours=47, minutes=59) < before - seen["cutoff"] <= timedelta(hours=48)
