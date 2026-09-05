"""Tests for HistoryManager — token estimation, tiered clearing, compaction, etc."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from assistant_runtime.services.history._manager import (
    COMPACTION_CACHE_KEY,
    SUMMARY_MARKER,
    TOOL_RESULT_PLACEHOLDER,
    HistoryManager,
)
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.models import CompactionResult


@pytest.fixture
def config():
    return HistoryConfig(token_budget=5000, retain_recent=2, protect_recent_tool_results=1)


@pytest.fixture
def mock_summarizer():
    summarizer = MagicMock()
    summarizer.summarize_structured = AsyncMock(
        return_value=CompactionResult(summary="Test summary")
    )
    return summarizer


@pytest.fixture
def manager(config, mock_summarizer):
    return HistoryManager(config, mock_summarizer)


def _make_user_msg(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _make_assistant_msg(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


def _make_tool_call_msg(tool_name: str = "test_tool", args: str = "{}") -> ModelResponse:
    return ModelResponse(
        parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=f"call_{tool_name}")]
    )


def _make_tool_result_msg(
    tool_name: str = "test_tool",
    content: str = "result",
    tool_call_id: str | None = None,
    **part_kwargs,
) -> ModelRequest:
    return ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name=tool_name,
                content=content,
                tool_call_id=tool_call_id or f"call_{tool_name}",
                **part_kwargs,
            )
        ]
    )


def _summary_messages(messages) -> list[ModelRequest]:
    return [
        m
        for m in messages
        if isinstance(m, ModelRequest)
        and any(
            isinstance(p, UserPromptPart)
            and isinstance(p.content, str)
            and SUMMARY_MARKER in p.content
            for p in m.parts
        )
    ]


class TestEstimateTokens:
    def test_empty_history(self, manager):
        assert manager._estimate_tokens([]) == 0

    def test_user_message(self, manager):
        # "hello" = 5 chars → 5 // 4 = 1
        history = [_make_user_msg("hello")]
        assert manager._estimate_tokens(history) == 1

    def test_known_string(self, manager):
        # 400 chars → 100 tokens
        content = "x" * 400
        history = [_make_user_msg(content)]
        assert manager._estimate_tokens(history) == 100

    def test_assistant_text(self, manager):
        # 80 chars → 20 tokens
        content = "a" * 80
        history = [_make_assistant_msg(content)]
        assert manager._estimate_tokens(history) == 20

    def test_tool_result(self, manager):
        # "result data" = 11 chars → 2 tokens
        history = [_make_tool_result_msg(content="result data")]
        assert manager._estimate_tokens(history) == 2

    def test_combined(self, manager):
        # "hello" (5) + "world" (5) = 10 chars → 2
        history = [_make_user_msg("hello"), _make_assistant_msg("world")]
        assert manager._estimate_tokens(history) == 2


class TestTieredToolClearing:
    def test_no_tool_results(self, manager):
        history = [_make_user_msg("hello"), _make_assistant_msg("hi")]
        result = manager._apply_tiered_tool_clearing(history)
        assert len(result) == 2

    def test_fewer_results_than_protect_count(self, manager):
        """With protect_count=1 and 1 tool result, nothing is cleared."""
        history = [
            _make_user_msg("q"),
            _make_tool_result_msg(content="important data"),
        ]
        result = manager._apply_tiered_tool_clearing(history)
        assert result is history

    def test_clears_older_preserves_recent(self, manager):
        """With protect_count=1, clears first tool result, keeps second."""
        history = [
            _make_tool_result_msg(tool_name="tool_a", content="old data"),
            _make_user_msg("middle"),
            _make_tool_result_msg(tool_name="tool_b", content="new data"),
        ]
        result = manager._apply_tiered_tool_clearing(history)

        assert result[0].parts[0].content == TOOL_RESULT_PLACEHOLDER
        assert result[0].parts[0].tool_name == "tool_a"
        assert result[1].parts[0].content == "middle"
        assert result[2].parts[0].content == "new data"

    def test_protect_zero_clears_all(self):
        config = HistoryConfig(protect_recent_tool_results=0)
        manager = HistoryManager(config, MagicMock())

        history = [
            _make_tool_result_msg(content="data1"),
            _make_tool_result_msg(content="data2"),
        ]
        result = manager._apply_tiered_tool_clearing(history)

        for msg in result:
            assert msg.parts[0].content == TOOL_RESULT_PLACEHOLDER

    def test_counts_individual_results_across_multi_tool_requests(self):
        """Protection counts tool results, not the messages that carry them."""
        config = HistoryConfig(protect_recent_tool_results=2)
        manager = HistoryManager(config, MagicMock())
        history = [
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="a", content="first", tool_call_id="a"),
                    ToolReturnPart(tool_name="b", content="second", tool_call_id="b"),
                ]
            ),
            _make_tool_result_msg(tool_name="c", content="third"),
        ]
        result = manager._apply_tiered_tool_clearing(history)
        assert [p.content for p in result[0].parts] == [TOOL_RESULT_PLACEHOLDER, "second"]
        assert result[1].parts[0].content == "third"

    def test_preserves_outcome_metadata_and_identity(self, manager):
        """Clearing replaces the content only; native fields survive."""
        history = [
            _make_tool_result_msg(
                tool_name="lookup",
                content="old data",
                outcome="interrupted",
                metadata={"provider": "x"},
            ),
            _make_tool_result_msg(tool_name="recent", content="new data"),
        ]
        result = manager._apply_tiered_tool_clearing(history)
        cleared = result[0].parts[0]
        original = history[0].parts[0]
        assert cleared.content == TOOL_RESULT_PLACEHOLDER
        assert cleared.outcome == "interrupted"
        assert cleared.metadata == {"provider": "x"}
        assert cleared.tool_call_id == original.tool_call_id
        assert cleared.timestamp == original.timestamp

    def test_does_not_mutate_original(self, manager):
        history = [
            _make_tool_result_msg(tool_name="old", content="original"),
            _make_tool_result_msg(tool_name="new", content="keep"),
        ]
        result = manager._apply_tiered_tool_clearing(history)
        assert history[0].parts[0].content == "original"
        assert result[1] is history[1]


class TestFrozenTail:
    async def test_frozen_results_are_never_cleared(self):
        config = HistoryConfig(protect_recent_tool_results=0)
        manager = HistoryManager(config, MagicMock())
        history = [
            _make_tool_result_msg(tool_name="old", content="old data"),
            _make_tool_result_msg(tool_name="current", content="current data"),
        ]
        result = manager._apply_tiered_tool_clearing(history, frozen_from=1)
        assert result[0].parts[0].content == TOOL_RESULT_PLACEHOLDER
        assert result[1] is history[1]

    async def test_frozen_messages_are_retained_verbatim_by_compaction(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)
        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("current"),
            _make_tool_call_msg("lookup"),
            _make_tool_result_msg("lookup"),
        ]
        result = await manager._compact(history, {}, config, frozen_from=2)
        assert result[-3:] == history[-3:]
        assert len(_summary_messages(result)) == 1

    async def test_nothing_before_the_current_run_returns_history(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)
        history = [_make_user_msg("current"), _make_assistant_msg("a" * 100)]
        result = await manager._compact(history, {}, config, frozen_from=0)
        assert result is history
        mock_summarizer.summarize_structured.assert_not_called()


class TestFormatTypedToDicts:
    def test_user_message(self):
        msgs = [_make_user_msg("hello")]
        result = HistoryManager._format_typed_to_dicts(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "hello"}

    def test_assistant_message(self):
        msgs = [_make_assistant_msg("hi there")]
        result = HistoryManager._format_typed_to_dicts(msgs)
        assert len(result) == 1
        assert result[0] == {"role": "assistant", "content": "hi there"}

    def test_tool_result_message(self):
        msgs = [_make_tool_result_msg(tool_name="get_status", content="active")]
        result = HistoryManager._format_typed_to_dicts(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Tool result (get_status): active" in result[0]["content"]

    def test_truncation(self):
        long_content = "x" * 2000
        msgs = [_make_user_msg(long_content)]
        result = HistoryManager._format_typed_to_dicts(msgs, truncation_limit=100)
        assert len(result[0]["content"]) < 200
        assert "truncated" in result[0]["content"]

    def test_tool_call_in_response(self):
        response = ModelResponse(
            parts=[
                TextPart(content="Let me check."),
                ToolCallPart(tool_name="get_agents", args='{"active": true}', tool_call_id="tc1"),
            ]
        )
        result = HistoryManager._format_typed_to_dicts([response])
        assert len(result) == 1
        assert "Let me check." in result[0]["content"]
        assert "[Called get_agents" in result[0]["content"]


class TestPrepareHistory:
    async def test_empty_history(self, manager):
        result, modified = await manager.prepare_history([], {})
        assert result == []
        assert modified is False

    async def test_within_budget_returns_same_list(self, manager):
        history = [_make_user_msg("short"), _make_assistant_msg("brief")]
        result, modified = await manager.prepare_history(history, {})
        assert modified is False
        assert result is history

    async def test_dangling_calls_are_left_to_the_native_pipeline(self, manager):
        history = [_make_user_msg("start"), _make_tool_call_msg("check_status")]
        result, _ = await manager.prepare_history(history, {})
        assert result is history

    async def test_over_budget_triggers_clearing(self):
        """With a tight budget but clearing brings it under."""
        config = HistoryConfig(
            token_budget=5000,
            retain_recent=2,
            protect_recent_tool_results=0,
        )
        summarizer = MagicMock()
        summarizer.summarize_structured = AsyncMock()
        manager = HistoryManager(config, summarizer)

        # Create history with tool results that exceed budget (5000 * 4 = 20000 chars)
        history = [
            _make_tool_result_msg(content="x" * 25000),  # 6250 tokens
            _make_user_msg("short"),  # small
        ]

        result, modified = await manager.prepare_history(history, {})
        assert modified is False  # Clearing alone was enough
        assert result[0].parts[0].content == TOOL_RESULT_PLACEHOLDER
        summarizer.summarize_structured.assert_not_called()

    async def test_over_budget_triggers_compaction(self, mock_summarizer):
        """When clearing isn't enough, triggers full compaction."""
        config = HistoryConfig(
            token_budget=5000,
            retain_recent=1,
            protect_recent_tool_results=0,
        )
        manager = HistoryManager(config, mock_summarizer)

        # 5000 tokens * 4 = 20000 chars, so we need > 20000 chars of non-tool content
        history = [
            _make_user_msg("x" * 8000),
            _make_assistant_msg("y" * 8000),
            _make_user_msg("z" * 8000),
            _make_assistant_msg("w" * 8000),
        ]

        session_context: dict = {}
        result, modified = await manager.prepare_history(history, session_context)

        assert modified is True
        mock_summarizer.summarize_structured.assert_called_once()
        # Compaction describes the model input only; working memory is a
        # separate per-turn extraction.
        assert "working_memory" not in session_context
        assert session_context[COMPACTION_CACHE_KEY]["prefix_count"] == 3
        assert len(_summary_messages(result)) == 1


class TestCompact:
    async def test_head_preservation(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)

        first_msg = _make_user_msg("What is the system status?")
        history = [
            first_msg,
            _make_assistant_msg("y" * 100),
            _make_user_msg("z" * 100),
            _make_assistant_msg("w" * 100),
        ]

        result = await manager._compact(history, {}, config)

        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "What is the system status?"
        assert SUMMARY_MARKER in result[1].parts[0].content
        # The head message is kept verbatim, so it is not summarized.
        summarized = mock_summarizer.summarize_structured.call_args[0][0]
        assert all("system status" not in m["content"] for m in summarized)

    async def test_summary_marker_present(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)

        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("last"),
        ]

        result = await manager._compact(history, {}, config)
        assert len(_summary_messages(result)) == 1

    async def test_retain_recent_respected(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=2)
        manager = HistoryManager(config, mock_summarizer)

        history = [
            _make_user_msg("old1"),
            _make_assistant_msg("old2"),
            _make_user_msg("recent1"),
            _make_assistant_msg("recent2"),
        ]

        result = await manager._compact(history, {}, config)

        assert result[-2].parts[0].content == "recent1"
        assert result[-1].parts[0].content == "recent2"

    async def test_retain_all_returns_unchanged(self, mock_summarizer):
        """When retain_recent >= len(history), no compaction."""
        config = HistoryConfig(token_budget=5000, retain_recent=10)
        manager = HistoryManager(config, mock_summarizer)

        history = [_make_user_msg("only"), _make_assistant_msg("two")]
        result = await manager._compact(history, {}, config)

        assert result is history

    async def test_compaction_boundary_skips_orphaned_tool_result(self, mock_summarizer):
        """When retain boundary lands on a tool result, shift it to avoid orphaning."""
        config = HistoryConfig(token_budget=5000, retain_recent=2)
        manager = HistoryManager(config, mock_summarizer)

        history = [
            _make_user_msg("old question"),
            _make_tool_call_msg("check_status"),
            _make_tool_result_msg("check_status"),
            _make_assistant_msg("Here are the results"),
        ]

        result = await manager._compact(history, {}, config)

        for idx, msg in enumerate(result):
            if not isinstance(msg, ModelRequest):
                continue
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    has_matching_call = any(
                        isinstance(p, ToolCallPart) and p.tool_call_id == part.tool_call_id
                        for m in result[:idx]
                        if isinstance(m, ModelResponse)
                        for p in m.parts
                    )
                    assert has_matching_call, f"Orphaned ToolReturnPart({part.tool_call_id})"

    async def test_cached_summary_is_reused_while_within_budget(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)
        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("second"),
            _make_assistant_msg("b" * 100),
        ]
        session_context: dict = {}
        first = await manager._compact(history, session_context, config)
        assert mock_summarizer.summarize_structured.call_count == 1

        # The next model request of the same turn has two more messages.
        longer = [*history, _make_tool_call_msg("lookup"), _make_tool_result_msg("lookup")]
        second = await manager._compact(longer, session_context, config)

        assert mock_summarizer.summarize_structured.call_count == 1
        assert _summary_messages(second)[0].parts[0].content == (
            _summary_messages(first)[0].parts[0].content
        )
        # Everything after the summarized prefix stays verbatim.
        assert second[-3:] == longer[-3:]

    async def test_cached_summary_is_extended_when_over_budget(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)
        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("second"),
        ]
        session_context: dict = {}
        await manager._compact(history, session_context, config)

        longer = [*history, _make_assistant_msg("b" * 21000), _make_user_msg("third")]
        await manager._compact(longer, session_context, config)

        assert mock_summarizer.summarize_structured.call_count == 2
        messages, existing_summary = mock_summarizer.summarize_structured.call_args[0]
        assert existing_summary == "Test summary"
        # Only the messages that left the retained window are summarized again.
        assert [m["content"][:6] for m in messages] == ["second", "bbbbbb"]
        assert session_context[COMPACTION_CACHE_KEY]["prefix_count"] == 4

    async def test_cached_summary_is_ignored_on_another_branch(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)
        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("second"),
        ]
        session_context: dict = {}
        await manager._compact(history, session_context, config)

        branch = [
            _make_user_msg("first"),
            _make_assistant_msg("a completely different answer"),
            _make_user_msg("second"),
        ]
        await manager._compact(branch, session_context, config)

        assert mock_summarizer.summarize_structured.call_count == 2
        assert mock_summarizer.summarize_structured.call_args[0][1] is None
