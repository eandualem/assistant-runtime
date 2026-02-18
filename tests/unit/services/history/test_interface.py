"""Tests for HistoryService lifecycle, delegation, and health_check."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.history.exceptions import CompactionError
from lovely_assistant.services.history.interface import HistoryService
from lovely_assistant.services.history.models import WorkingMemory


@pytest.fixture
def config():
    return HistoryConfig()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def service(config, mock_llm):
    return HistoryService(config=config, llm_service=mock_llm)


class TestLifecycle:
    async def test_start_creates_manager(self, service):
        await service.start()
        assert service._started is True
        assert service._manager is not None

    async def test_stop_clears_state(self, service):
        await service.start()
        await service.stop()
        assert service._started is False
        assert service._manager is None

    async def test_health_check_before_start(self, service):
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_check_after_start(self, service):
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True
        assert health["token_budget"] == 100_000

    async def test_health_check_after_stop(self, service):
        await service.start()
        await service.stop()
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_check_includes_token_budget(self):
        config = HistoryConfig(token_budget=50_000)
        service = HistoryService(config=config, llm_service=MagicMock())
        health = await service.health_check()
        assert health["token_budget"] == 50_000


class TestPrepareHistory:
    async def test_raises_if_not_started(self, service):
        with pytest.raises(CompactionError, match="not started"):
            await service.prepare_history([], {})

    async def test_delegates_to_manager(self, service):
        await service.start()

        # Mock the manager's prepare_history
        service._manager.prepare_history = AsyncMock(return_value=([], False))

        result, modified = await service.prepare_history([], {})
        assert result == []
        assert modified is False
        service._manager.prepare_history.assert_called_once()

    async def test_passes_is_continuation(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(return_value=([], False))

        await service.prepare_history([], {}, is_continuation=True)

        call_kwargs = service._manager.prepare_history.call_args.kwargs
        assert call_kwargs["is_continuation"] is True

    async def test_wraps_unexpected_errors(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(side_effect=RuntimeError("unexpected"))

        with pytest.raises(CompactionError, match="History preparation failed"):
            await service.prepare_history([], {})

    async def test_reraises_compaction_errors(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(side_effect=CompactionError("specific issue"))

        with pytest.raises(CompactionError, match="specific issue"):
            await service.prepare_history([], {})


class TestExtractMemoryDelta:
    async def test_raises_if_not_started(self, service):
        with pytest.raises(CompactionError, match="not started"):
            await service.extract_memory_delta(WorkingMemory(), [], turn_number=1)

    async def test_delegates_to_summarizer(self, service):
        await service.start()

        expected_wm = WorkingMemory(active_goal="Updated")
        service._manager._summarizer.extract_memory_delta = AsyncMock(return_value=expected_wm)

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "test"}]
        result = await service.extract_memory_delta(wm, messages, turn_number=5)

        assert result.active_goal == "Updated"
        service._manager._summarizer.extract_memory_delta.assert_called_once_with(wm, messages, 5)
