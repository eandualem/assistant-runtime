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
    SUMMARY_MARKER,
    TOOL_RESULT_PLACEHOLDER,
    HistoryManager,
)
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.models import CompactionResult, WorkingMemory


@pytest.fixture
def config():
    return HistoryConfig(token_budget=5000, retain_recent=2, protect_recent_tool_results=1)


@pytest.fixture
def mock_summarizer():
    summarizer = MagicMock()
    summarizer.summarize_structured = AsyncMock(
        return_value=CompactionResult(
            summary="Test summary",
            working_memory=WorkingMemory(active_goal="Test goal"),
        )
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
    tool_name: str = "test_tool", content: str = "result", tool_call_id: str | None = None
) -> ModelRequest:
    return ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name=tool_name,
                content=content,
                tool_call_id=tool_call_id or f"call_{tool_name}",
            )
        ]
    )


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
        # Content should be unchanged
        assert result[1].parts[0].content == "important data"

    def test_clears_older_preserves_recent(self, manager):
        """With protect_count=1, clears first tool result, keeps second."""
        history = [
            _make_tool_result_msg(tool_name="tool_a", content="old data"),
            _make_user_msg("middle"),
            _make_tool_result_msg(tool_name="tool_b", content="new data"),
        ]
        result = manager._apply_tiered_tool_clearing(history)

        # First tool result should be cleared
        assert result[0].parts[0].content == TOOL_RESULT_PLACEHOLDER
        assert result[0].parts[0].tool_name == "tool_a"
        # User message unchanged
        assert result[1].parts[0].content == "middle"
        # Second tool result preserved
        assert result[2].parts[0].content == "new data"

    def test_protect_zero_clears_all(self):
        config = HistoryConfig(protect_recent_tool_results=0)
        summarizer = MagicMock()
        manager = HistoryManager(config, summarizer)

        history = [
            _make_tool_result_msg(content="data1"),
            _make_tool_result_msg(content="data2"),
        ]
        result = manager._apply_tiered_tool_clearing(history)

        for msg in result:
            assert msg.parts[0].content == TOOL_RESULT_PLACEHOLDER

    def test_does_not_mutate_original(self, manager):
        history = [
            _make_tool_result_msg(content="original"),
            _make_tool_result_msg(content="keep"),
        ]
        manager._apply_tiered_tool_clearing(history)
        # Original should be unchanged
        assert history[0].parts[0].content == "original"


class TestResolveDanglingToolCalls:
    def test_no_history(self):
        result = HistoryManager._resolve_dangling_tool_calls([])
        assert result == []

    def test_last_message_not_response(self):
        history = [_make_user_msg("hello")]
        result = HistoryManager._resolve_dangling_tool_calls(history)
        assert len(result) == 1

    def test_response_without_tool_calls(self):
        history = [_make_assistant_msg("just text")]
        result = HistoryManager._resolve_dangling_tool_calls(history)
        assert len(result) == 1

    def test_adds_synthetic_result(self):
        history = [_make_tool_call_msg("check_status")]
        result = HistoryManager._resolve_dangling_tool_calls(history)

        assert len(result) == 2
        synthetic = result[1]
        assert isinstance(synthetic, ModelRequest)
        assert isinstance(synthetic.parts[0], ToolReturnPart)
        assert synthetic.parts[0].tool_name == "check_status"
        assert "interrupted" in synthetic.parts[0].content

    def test_handles_multiple_tool_calls(self):
        response = ModelResponse(
            parts=[
                ToolCallPart(tool_name="tool_a", args="{}", tool_call_id="a"),
                ToolCallPart(tool_name="tool_b", args="{}", tool_call_id="b"),
            ]
        )
        history = [response]
        result = HistoryManager._resolve_dangling_tool_calls(history)

        assert len(result) == 2
        synthetic = result[1]
        assert len(synthetic.parts) == 2
        assert synthetic.parts[0].tool_name == "tool_a"
        assert synthetic.parts[1].tool_name == "tool_b"

    def test_does_not_mutate_original(self):
        history = [_make_tool_call_msg("test")]
        original_len = len(history)
        HistoryManager._resolve_dangling_tool_calls(history)
        assert len(history) == original_len

    def test_inserts_synthetic_result_before_followup_user_message(self):
        history = [
            _make_user_msg("start"),
            _make_tool_call_msg("navigate"),
            _make_user_msg("edited branch"),
        ]

        result = HistoryManager._resolve_dangling_tool_calls(history)

        assert len(result) == 4
        assert result[0].parts[0].content == "start"
        assert isinstance(result[1], ModelResponse)
        assert isinstance(result[2], ModelRequest)
        assert isinstance(result[2].parts[0], ToolReturnPart)
        assert result[2].parts[0].tool_name == "navigate"
        assert result[3].parts[0].content == "edited branch"

    def test_does_not_add_synthetic_result_when_tool_output_exists_before_followup(self):
        history = [
            _make_user_msg("start"),
            _make_tool_call_msg("navigate"),
            _make_tool_result_msg("navigate", "done", tool_call_id="call_navigate"),
            _make_user_msg("continue"),
        ]

        result = HistoryManager._resolve_dangling_tool_calls(history)

        assert len(result) == 4
        assert isinstance(result[2], ModelRequest)
        assert result[2].parts[0].content == "done"


class TestIsSummaryMessage:
    def test_summary_message(self):
        msg = _make_user_msg(f"{SUMMARY_MARKER}\nSome summary content")
        assert HistoryManager._is_summary_message(msg) is True

    def test_regular_user_message(self):
        msg = _make_user_msg("How are the agents doing?")
        assert HistoryManager._is_summary_message(msg) is False

    def test_assistant_response(self):
        msg = _make_assistant_msg("some response")
        assert HistoryManager._is_summary_message(msg) is False


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

    async def test_within_budget_no_changes(self, manager):
        history = [_make_user_msg("short"), _make_assistant_msg("brief")]
        result, modified = await manager.prepare_history(history, {})
        assert modified is False
        assert len(result) == 2

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
        # After clearing, the tool result is replaced with short placeholder
        # Should fit within budget now
        assert modified is False  # Clearing alone was enough
        assert result[0].parts[0].content == TOOL_RESULT_PLACEHOLDER

    async def test_over_budget_triggers_compaction(self, mock_summarizer):
        """When clearing isn't enough, triggers full compaction."""
        config = HistoryConfig(
            token_budget=5000,
            retain_recent=1,
            protect_recent_tool_results=0,
        )
        manager = HistoryManager(config, mock_summarizer)

        # Create history large enough that even after clearing we're over budget
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
        # Summarizer should have been called
        mock_summarizer.summarize_structured.assert_called_once()
        # Working memory should be set in session context
        assert "working_memory" in session_context

    async def test_continuation_excludes_pending_tool_from_resolution(self, manager):
        """Continuation resolves all dangling EXCEPT the excluded pending tool."""
        history = [
            _make_user_msg("start"),
            _make_tool_call_msg("ui_navigate"),  # Dangling — will be resolved by host
        ]
        result, _ = await manager.prepare_history(
            history,
            {},
            is_continuation=True,
            exclude_tool_call_ids={"call_ui_navigate"},
        )
        # Should NOT have added a synthetic tool result for the excluded tool
        assert len(result) == 2

    async def test_continuation_resolves_non_excluded_dangling(self, manager):
        """Continuation still resolves dangling tools that are NOT excluded."""
        history = [
            _make_user_msg("start"),
            _make_tool_call_msg("old_stale_tool"),  # Dangling from earlier turn
            _make_user_msg("next message"),
            _make_tool_call_msg("ui_navigate"),  # Current pending tool
        ]
        result, _ = await manager.prepare_history(
            history,
            {},
            is_continuation=True,
            exclude_tool_call_ids={"call_ui_navigate"},
        )
        # old_stale_tool should get a synthetic result, ui_navigate should not
        assert len(result) == 5  # original 4 + 1 synthetic for old_stale_tool

    async def test_non_continuation_resolves_dangling(self, manager):
        """Without is_continuation, dangling tool calls get synthetic results."""
        history = [
            _make_user_msg("start"),
            _make_tool_call_msg("check_status"),
        ]
        result, _ = await manager.prepare_history(history, {})
        # Should have added synthetic tool result
        assert len(result) == 3
        assert isinstance(result[2], ModelRequest)
        assert isinstance(result[2].parts[0], ToolReturnPart)


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

        # First message should be preserved
        assert isinstance(result[0], ModelRequest)
        assert result[0].parts[0].content == "What is the system status?"
        # Second should be summary
        assert SUMMARY_MARKER in result[1].parts[0].content

    async def test_summary_marker_present(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)

        history = [
            _make_user_msg("first"),
            _make_assistant_msg("a" * 100),
            _make_user_msg("last"),
        ]

        result = await manager._compact(history, {}, config)

        # Find the summary message
        summary_msgs = [m for m in result if HistoryManager._is_summary_message(m)]
        assert len(summary_msgs) == 1

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

        # Last two messages should be preserved verbatim
        assert result[-2].parts[0].content == "recent1"
        assert result[-1].parts[0].content == "recent2"

    async def test_working_memory_set_in_context(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)

        history = [
            _make_user_msg("first"),
            _make_assistant_msg("response"),
            _make_user_msg("second"),
        ]

        session_context: dict = {}
        await manager._compact(history, session_context, config)

        assert "working_memory" in session_context
        assert session_context["working_memory"]["active_goal"] == "Test goal"

    async def test_retain_all_returns_unchanged(self, mock_summarizer):
        """When retain_recent >= len(history), no compaction."""
        config = HistoryConfig(token_budget=5000, retain_recent=10)
        manager = HistoryManager(config, mock_summarizer)

        history = [_make_user_msg("only"), _make_assistant_msg("two")]
        result = await manager._compact(history, {}, config)

        assert result == history

    async def test_compaction_boundary_skips_orphaned_tool_result(self, mock_summarizer):
        """When retain boundary lands on a tool result, shift it to avoid orphaning."""
        config = HistoryConfig(token_budget=5000, retain_recent=2)
        manager = HistoryManager(config, mock_summarizer)

        # History: user → assistant(tool_call) → tool_result → assistant(text)
        # With retain_recent=2, the naive split puts tool_result + assistant(text) in to_keep.
        # That orphans the tool_result (tool_call is summarized away).
        history = [
            _make_user_msg("old question"),
            _make_tool_call_msg("check_status"),  # index 1 — ModelResponse with ToolCallPart
            _make_tool_result_msg("check_status"),  # index 2 — ModelRequest with ToolReturnPart
            _make_assistant_msg("Here are the results"),  # index 3
        ]

        result = await manager._compact(history, {}, config)

        # The tool_result should NOT be the first message in to_keep.
        # The boundary should shift so only the assistant text is retained.
        # Result should be: [first_user, summary, assistant_text] or similar.
        for msg in result:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    # No orphaned ToolReturnPart should appear without a preceding ToolCallPart
                    if isinstance(part, ToolReturnPart):
                        # Find the preceding messages and verify there's a matching ToolCallPart
                        idx = result.index(msg)
                        preceding = result[:idx]
                        has_matching_call = any(
                            isinstance(p, ToolCallPart) and p.tool_call_id == part.tool_call_id
                            for m in preceding
                            if isinstance(m, ModelResponse)
                            for p in m.parts
                        )
                        assert has_matching_call, (
                            f"Orphaned ToolReturnPart({part.tool_call_id}) with no matching ToolCallPart"
                        )

    async def test_existing_summary_passed_to_summarizer(self, mock_summarizer):
        config = HistoryConfig(token_budget=5000, retain_recent=1)
        manager = HistoryManager(config, mock_summarizer)

        summary_msg = _make_user_msg(f"{SUMMARY_MARKER}\nPrevious summary content")
        history = [
            summary_msg,
            _make_user_msg("new question"),
            _make_assistant_msg("answer"),
            _make_user_msg("follow up"),
        ]

        await manager._compact(history, {}, config)

        # Verify existing summary was passed
        call_args = mock_summarizer.summarize_structured.call_args
        assert call_args[0][1] == "Previous summary content"
