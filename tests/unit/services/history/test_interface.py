"""Tests for HistoryService lifecycle, delegation, and health_check."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.exceptions import CompactionError
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.history.models import HistoryPreparationResult, WorkingMemory


def _ctx(run_id: str = "run-1"):
    return SimpleNamespace(run_id=run_id)


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

    async def test_set_runtime_settings_propagates_to_summarizer(self, service):
        await service.start()
        runtime = MagicMock()

        service.set_runtime_settings(runtime)

        assert service._runtime_settings is runtime
        assert service._manager is not None
        assert service._manager._summarizer._runtime_settings is runtime


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
        service._manager._summarizer.extract_memory_delta.assert_called_once_with(
            wm, messages, 5, usage=None
        )


class TestProcessor:
    async def test_raises_if_not_started(self, service):
        with pytest.raises(CompactionError, match="not started"):
            service.processor({})

    async def test_disabled_compaction_has_no_processor(self, mock_llm):
        service = HistoryService(HistoryConfig(compaction_enabled=False), llm_service=mock_llm)
        await service.start()
        assert service.processor({}) is None

    async def test_capability_is_native_process_history(self, service):
        await service.start()
        assert isinstance(service.processor({}).capability(), ProcessHistory)

    async def test_process_records_result(self, service):
        await service.start()
        user_msg = ModelRequest(parts=[UserPromptPart(content="Hello")])
        assistant_msg = ModelResponse(parts=[TextPart(content="Hi there")])
        service._manager.prepare_history = AsyncMock(return_value=([user_msg], True))
        service._manager._estimate_tokens = MagicMock(return_value=500)
        processor = service.processor({})
        assert processor.result is None

        prepared = await processor.process(_ctx(), [user_msg, assistant_msg])

        assert prepared == [user_msg]
        result = processor.result
        assert isinstance(result, HistoryPreparationResult)
        assert result.was_compacted is True
        assert result.message_count == 1
        assert result.estimated_tokens == 500
        assert result.compacted_from == 2
        assert result.message_summaries[0]["content_preview"] == "Hello"

    async def test_process_freezes_the_current_run(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(return_value=([], False))
        service._manager._estimate_tokens = MagicMock(return_value=0)
        older = ModelRequest(parts=[UserPromptPart(content="old")], run_id="run-0")
        current = ModelRequest(parts=[UserPromptPart(content="new")], run_id="run-1")

        await service.processor({}).process(_ctx("run-1"), [older, older, current])

        assert service._manager.prepare_history.call_args.kwargs["frozen_from"] == 2

    async def test_process_passes_session_context(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(return_value=([], False))
        service._manager._estimate_tokens = MagicMock(return_value=0)
        context: dict = {}

        await service.processor(context).process(_ctx(), [])

        assert service._manager.prepare_history.call_args[0][1] is context

    async def test_wraps_unexpected_errors(self, service):
        await service.start()
        service._manager.prepare_history = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(CompactionError, match="History preparation failed"):
            await service.processor({}).process(_ctx(), [])


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
