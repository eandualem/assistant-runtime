"""Tests for HistoryService lifecycle, delegation, and health_check."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.history.exceptions import CompactionError
from lovely_assistant.services.history.interface import HistoryService
from lovely_assistant.services.history.models import HistoryPreparationResult, WorkingMemory


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


class TestPrepareHistoryWithMetadata:
    async def test_raises_if_not_started(self, service):
        with pytest.raises(CompactionError, match="not started"):
            await service.prepare_history_with_metadata([], {})

    async def test_returns_history_preparation_result(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(return_value=([], False))
        service._manager._estimate_tokens = MagicMock(return_value=0)

        result = await service.prepare_history_with_metadata([], {})
        assert isinstance(result, HistoryPreparationResult)
        assert result.history == []
        assert result.was_compacted is False
        assert result.message_count == 0
        assert result.estimated_tokens == 0
        assert result.compacted_from == 0

    async def test_compacted_result(self, service):
        await service.start()
        # Simulate compaction: prepare_history returns was_compacted=True with fewer messages
        service._manager.prepare_history = AsyncMock(return_value=([MagicMock()], True))
        service._manager._estimate_tokens = MagicMock(return_value=500)

        # Pass 5 messages, expect compacted_from=5
        history = [MagicMock() for _ in range(5)]
        result = await service.prepare_history_with_metadata(history, {})
        assert result.was_compacted is True
        assert result.message_count == 1
        assert result.estimated_tokens == 500
        assert result.compacted_from == 5

    async def test_passes_is_continuation(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(return_value=([], False))
        service._manager._estimate_tokens = MagicMock(return_value=0)

        await service.prepare_history_with_metadata([], {}, is_continuation=True)
        call_kwargs = service._manager.prepare_history.call_args.kwargs
        assert call_kwargs["is_continuation"] is True

    async def test_wraps_unexpected_errors(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(CompactionError, match="History preparation failed"):
            await service.prepare_history_with_metadata([], {})

    async def test_includes_message_summaries(self, service):
        await service.start()

        user_msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
        assistant_msg = ModelResponse(parts=[TextPart(content="Hi there")])
        prepared = [user_msg, assistant_msg]

        service._manager.prepare_history = AsyncMock(return_value=(prepared, False))
        service._manager._estimate_tokens = MagicMock(return_value=50)

        result = await service.prepare_history_with_metadata(prepared, {})
        assert len(result.message_summaries) == 2
        assert result.message_summaries[0]["role"] == "user"
        assert result.message_summaries[0]["content_preview"] == "Hello"
        assert result.message_summaries[1]["role"] == "assistant"
        assert result.message_summaries[1]["content_preview"] == "Hi there"


class TestSummarizeMessages:
    def test_empty_list(self):
        result = HistoryService._summarize_messages([])
        assert result == []

    def test_user_message(self):
        msg = ModelRequest(parts=[UserPromptPart(content="What is 2+2?")])
        result = HistoryService._summarize_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content_preview"] == "What is 2+2?"
        assert result[0]["char_count"] == 12
        assert result[0]["part_count"] == 1

    def test_assistant_message(self):
        msg = ModelResponse(parts=[TextPart(content="The answer is 4.")])
        result = HistoryService._summarize_messages([msg])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content_preview"] == "The answer is 4."
        assert result[0]["char_count"] == 16
        assert result[0]["part_count"] == 1

    def test_multi_part_message(self):
        msg = ModelRequest(
            parts=[
                UserPromptPart(content="Part one."),
                UserPromptPart(content="Part two."),
            ]
        )
        result = HistoryService._summarize_messages([msg])
        assert result[0]["part_count"] == 2
        assert result[0]["char_count"] == len("Part one.\nPart two.")
        assert "Part one." in result[0]["content_preview"]
        assert "Part two." in result[0]["content_preview"]

    def test_preview_truncation(self):
        long_content = "x" * 1000
        msg = ModelRequest(parts=[UserPromptPart(content=long_content)])
        result = HistoryService._summarize_messages([msg], preview_limit=100)
        assert len(result[0]["content_preview"]) == 100
        assert result[0]["char_count"] == 1000

    def test_system_prompt_part_excluded_from_content(self):
        msg = ModelRequest(parts=[SystemPromptPart(content="System instruction")])
        result = HistoryService._summarize_messages([msg])
        assert result[0]["role"] == "user"
        # SystemPromptPart has .content, so it should be included
        assert result[0]["char_count"] > 0

    def test_mixed_conversation(self):
        messages = [
            ModelRequest(parts=[UserPromptPart(content="Hi")]),
            ModelResponse(parts=[TextPart(content="Hello!")]),
            ModelRequest(parts=[UserPromptPart(content="How are you?")]),
            ModelResponse(parts=[TextPart(content="I'm great, thanks!")]),
        ]
        result = HistoryService._summarize_messages(messages)
        assert len(result) == 4
        roles = [r["role"] for r in result]
        assert roles == ["user", "assistant", "user", "assistant"]
