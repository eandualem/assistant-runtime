"""Tests for HistorySummarizer — all LLM interactions mocked."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.services.history._summarizer import HistorySummarizer
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.models import (
    CompactionResult,
    MemoryDelta,
    MemoryDeltaResult,
    MemoryEntry,
    MemoryOperation,
    WorkingMemory,
)


@pytest.fixture
def config():
    return HistoryConfig()


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def summarizer(config, mock_llm):
    return HistorySummarizer(config, mock_llm)


def _mock_agent_run(output):
    """Create a mock agent whose run() returns a result with .output set."""
    mock_result = MagicMock()
    mock_result.output = output
    mock_agent = MagicMock()
    mock_agent.run = AsyncMock(return_value=mock_result)
    return mock_agent


class TestSummarizeStructured:
    async def test_empty_messages_returns_default(self, summarizer):
        result = await summarizer.summarize_structured([])
        assert isinstance(result, CompactionResult)
        assert result.summary == ""

    async def test_empty_messages_preserves_existing_summary(self, summarizer):
        result = await summarizer.summarize_structured([], existing_summary="Prior context")
        assert result.summary == "Prior context"

    async def test_calls_llm_with_correct_output_type(self, summarizer, mock_llm):
        expected = CompactionResult(
            summary="Agent checked tmux sessions.",
            user_goal="Monitor agents",
        )
        mock_llm.build_agent.return_value = _mock_agent_run(expected)

        messages = [
            {"role": "user", "content": "Check agent status"},
            {"role": "assistant", "content": "All agents are running."},
        ]
        result = await summarizer.summarize_structured(messages)

        # Verify build_agent was called with CompactionResult output_type
        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["output_type"] is CompactionResult
        assert call_kwargs["system_prompt"]  # Non-empty
        assert result.summary == "Agent checked tmux sessions."
        assert result.user_goal == "Monitor agents"

    async def test_passes_existing_summary(self, summarizer, mock_llm):
        expected = CompactionResult(summary="Updated summary.")
        mock_llm.build_agent.return_value = _mock_agent_run(expected)

        messages = [{"role": "user", "content": "Follow up question"}]
        await summarizer.summarize_structured(messages, existing_summary="Prior context")

        # The prompt should include the existing summary
        agent = mock_llm.build_agent.return_value
        call_args = agent.run.call_args[0][0]  # First positional arg = prompt
        assert "Prior context" in call_args

    async def test_fallback_on_llm_failure(self, summarizer, mock_llm):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API error"))
        mock_llm.build_agent.return_value = mock_agent

        messages = [
            {"role": "user", "content": "Check status"},
            {"role": "assistant", "content": "Response"},
        ]
        result = await summarizer.summarize_structured(messages)

        assert isinstance(result, CompactionResult)
        assert "Previous conversation:" in result.summary
        assert result.current_state == "Unknown (summarization failed)"

    async def test_fallback_includes_existing_summary(self, summarizer, mock_llm):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API error"))
        mock_llm.build_agent.return_value = mock_agent

        messages = [{"role": "user", "content": "test"}]
        result = await summarizer.summarize_structured(messages, existing_summary="Old summary")

        assert "Old summary" in result.summary

    async def test_no_override_uses_llm_summarization_default(self, mock_llm):
        mock_llm.resolve_summarization_model.return_value = "anthropic:claude-haiku-4-5"
        summarizer = HistorySummarizer(HistoryConfig(), mock_llm)
        mock_llm.build_agent.return_value = _mock_agent_run(CompactionResult(summary="s"))

        await summarizer.summarize_structured([{"role": "user", "content": "hello"}])

        assert mock_llm.build_agent.call_args.kwargs["model"] == "anthropic:claude-haiku-4-5"

    async def test_model_override_passed_to_build_agent(self, mock_llm):
        config = HistoryConfig(summarization_model="openai:gpt-4o-mini")
        summarizer = HistorySummarizer(config, mock_llm)

        expected = CompactionResult(summary="test")
        mock_llm.build_agent.return_value = _mock_agent_run(expected)

        messages = [{"role": "user", "content": "hello"}]
        await summarizer.summarize_structured(messages)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o-mini"


class TestExtractMemoryDelta:
    async def test_empty_messages_returns_current(self, summarizer):
        wm = WorkingMemory(active_goal="Test")
        result = await summarizer.extract_memory_delta(wm, [], turn_number=1)
        assert result.active_goal == "Test"

    async def test_applies_deltas_from_llm(self, summarizer, mock_llm):
        delta_result = MemoryDeltaResult(
            active_goal="Updated goal",
            progress="Made progress",
            next_steps="Next step",
            deltas=[
                MemoryDelta(
                    operation=MemoryOperation.ADD,
                    entry=MemoryEntry(
                        content="New insight",
                        category="agent_behavior",
                        confidence=0.9,
                    ),
                )
            ],
        )
        mock_llm.build_agent.return_value = _mock_agent_run(delta_result)

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "Check status"}]

        result = await summarizer.extract_memory_delta(wm, messages, turn_number=5)

        assert result.active_goal == "Updated goal"
        assert result.progress == "Made progress"
        assert len(result.entries) == 1
        assert result.entries[0].content == "New insight"
        assert result.entries[0].created_turn == 5

    async def test_preserves_existing_entries(self, summarizer, mock_llm):
        existing_entry = MemoryEntry(
            id="existing-id",
            content="Existing insight",
            category="backbone_pattern",
            created_turn=1,
        )
        delta_result = MemoryDeltaResult(
            active_goal="goal",
            deltas=[],  # No changes to entries
        )
        mock_llm.build_agent.return_value = _mock_agent_run(delta_result)

        wm = WorkingMemory(entries=[existing_entry])
        messages = [{"role": "user", "content": "hello"}]

        result = await summarizer.extract_memory_delta(wm, messages, turn_number=5)

        assert len(result.entries) == 1
        assert result.entries[0].id == "existing-id"

    async def test_fallback_on_failure_returns_current(self, summarizer, mock_llm):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=RuntimeError("API error"))
        mock_llm.build_agent.return_value = mock_agent

        wm = WorkingMemory(active_goal="Original goal")
        messages = [{"role": "user", "content": "test"}]

        result = await summarizer.extract_memory_delta(wm, messages, turn_number=1)

        # Should return original unchanged
        assert result.active_goal == "Original goal"

    async def test_retries_on_first_failure(self, summarizer, mock_llm):
        """On first failure, retries and succeeds on second attempt."""
        delta_result = MemoryDeltaResult(active_goal="Success")

        mock_result = MagicMock()
        mock_result.output = delta_result
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=[RuntimeError("First fail"), mock_result])
        mock_llm.build_agent.return_value = mock_agent

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "test"}]

        result = await summarizer.extract_memory_delta(wm, messages, turn_number=1)

        assert result.active_goal == "Success"
        assert mock_agent.run.call_count == 2

    async def test_working_memory_model_override(self, mock_llm):
        config = HistoryConfig(working_memory_model="openai:gpt-4o")
        summarizer = HistorySummarizer(config, mock_llm)

        delta_result = MemoryDeltaResult(active_goal="test")
        mock_llm.build_agent.return_value = _mock_agent_run(delta_result)

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "hello"}]
        await summarizer.extract_memory_delta(wm, messages, turn_number=1)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o"

    async def test_falls_back_to_summarization_model(self, mock_llm):
        config = HistoryConfig(
            summarization_model="openai:gpt-4o-mini",
            working_memory_model=None,
        )
        summarizer = HistorySummarizer(config, mock_llm)

        delta_result = MemoryDeltaResult(active_goal="test")
        mock_llm.build_agent.return_value = _mock_agent_run(delta_result)

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "hello"}]
        await summarizer.extract_memory_delta(wm, messages, turn_number=1)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o-mini"


class TestFormatMessages:
    def test_formats_user_message(self, summarizer):
        messages = [{"role": "user", "content": "Hello"}]
        result = summarizer._format_messages_for_summarization(messages)
        assert "[USER]: Hello" in result

    def test_formats_assistant_message(self, summarizer):
        messages = [{"role": "assistant", "content": "Hi there"}]
        result = summarizer._format_messages_for_summarization(messages)
        assert "[ASSISTANT]: Hi there" in result

    def test_truncates_long_content(self):
        config = HistoryConfig(message_truncation_limit=100)
        mock_llm = MagicMock()
        summarizer = HistorySummarizer(config, mock_llm)

        messages = [{"role": "user", "content": "x" * 500}]
        result = summarizer._format_messages_for_summarization(messages)
        assert "truncated" in result
        assert len(result) < 500


class TestRuntimeSettingsOverride:
    async def test_summarize_uses_runtime_summarization_model(self, config, mock_llm):
        """When runtime_settings overrides summarization_model, build_agent uses it."""
        mock_rs = MagicMock()
        mock_rs.get = MagicMock(
            side_effect=lambda field, default: (
                "openai:gpt-4o-mini" if field == "summarization_model" else default
            )
        )
        summarizer = HistorySummarizer(config, mock_llm, runtime_settings=mock_rs)

        expected = CompactionResult(summary="test")
        mock_llm.build_agent.return_value = _mock_agent_run(expected)

        messages = [{"role": "user", "content": "hello"}]
        await summarizer.summarize_structured(messages)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o-mini"
        mock_rs.get.assert_any_call("summarization_model", config.summarization_model)

    async def test_extract_memory_uses_runtime_working_memory_model(self, config, mock_llm):
        """When runtime_settings overrides working_memory_model, build_agent uses it."""
        mock_rs = MagicMock()
        mock_rs.get = MagicMock(
            side_effect=lambda field, default: (
                "openai:gpt-4o" if field == "working_memory_model" else default
            )
        )
        summarizer = HistorySummarizer(config, mock_llm, runtime_settings=mock_rs)

        delta_result = MemoryDeltaResult(active_goal="test")
        mock_llm.build_agent.return_value = _mock_agent_run(delta_result)

        wm = WorkingMemory()
        messages = [{"role": "user", "content": "hello"}]
        await summarizer.extract_memory_delta(wm, messages, turn_number=1)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o"
        mock_rs.get.assert_any_call("working_memory_model", config.working_memory_model)

    async def test_no_runtime_settings_uses_config(self, mock_llm):
        """When runtime_settings is None, config defaults are used."""
        config = HistoryConfig(summarization_model="openai:gpt-4o-mini")
        summarizer = HistorySummarizer(config, mock_llm, runtime_settings=None)

        expected = CompactionResult(summary="test")
        mock_llm.build_agent.return_value = _mock_agent_run(expected)

        messages = [{"role": "user", "content": "hello"}]
        await summarizer.summarize_structured(messages)

        call_kwargs = mock_llm.build_agent.call_args.kwargs
        assert call_kwargs["model"] == "openai:gpt-4o-mini"


class TestFallbackCompactionResult:
    def test_basic_fallback(self, summarizer):
        messages = [
            {"role": "user", "content": "Check agent status"},
            {"role": "assistant", "content": "All running."},
        ]
        result = summarizer._create_fallback_compaction_result(messages)

        assert "Previous conversation:" in result.summary
        assert "2 messages" in result.summary
        assert result.user_goal == "Check agent status"
        assert result.current_state == "Unknown (summarization failed)"

    def test_includes_existing_summary(self, summarizer):
        messages = [{"role": "user", "content": "test"}]
        result = summarizer._create_fallback_compaction_result(
            messages, existing_summary="Prior context"
        )
        assert "Prior context" in result.summary

    def test_empty_messages(self, summarizer):
        result = summarizer._create_fallback_compaction_result([])
        assert "0 messages" in result.summary
