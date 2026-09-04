"""Conversation history management with context window optimization.

Internal module — only accessed through HistoryService (interface.py).
Translated from an earlier assistant implementation's history/manager.py.

Manages conversation history using token-budget-based compaction triggers,
tiered tool result clearing, head preservation, and structured working memory
extraction.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from loguru import logger
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from assistant_runtime.services.history._summarizer import HistorySummarizer
from assistant_runtime.services.history.config import HistoryConfig

SUMMARY_MARKER = "[CONVERSATION SUMMARY]"
TOOL_RESULT_PLACEHOLDER = "[Tool result cleared to save context space]"
_DEFAULT_MESSAGE_TRUNCATION_LIMIT = 1000


class HistoryManager:
    """Manages conversation history with token-budget-based compaction.

    Two-stage compaction strategy:
    1. Tiered tool result clearing — replaces older tool results with placeholders
    2. Summarization compaction — LLM-backed summarization of older messages
    """

    def __init__(self, config: HistoryConfig, summarizer: HistorySummarizer) -> None:
        self.config = config
        self._summarizer = summarizer

    def set_runtime_settings(self, runtime_settings: Any | None) -> None:
        """Forward runtime settings to summarizer dependency."""
        self._summarizer.set_runtime_settings(runtime_settings)

    async def prepare_history(
        self,
        history: list[ModelMessage],
        session_context: dict[str, Any],
        *,
        is_continuation: bool = False,
        exclude_tool_call_ids: set[str] | None = None,
        config_override: HistoryConfig | None = None,
    ) -> tuple[list[ModelMessage], bool]:
        """Prepare history for the agent loop.

        Checks the token budget first. If over budget, applies tiered
        tool result clearing, then full summarization compaction if still over.
        """
        config = config_override or self.config

        if not history:
            return history, False

        # Resolve dangling tool calls. For continuations, exclude the pending
        # tool_call_id (DeferredToolResults will provide its result) but still
        # resolve any other dangling calls from earlier turns.
        history = self._resolve_dangling_tool_calls(history, exclude_ids=exclude_tool_call_ids)

        estimated_tokens = self._estimate_tokens(history)

        if estimated_tokens <= config.token_budget:
            logger.debug(
                f"[HISTORY] Token estimate {estimated_tokens} within budget "
                f"{config.token_budget}, no compaction needed"
            )
            return history, False

        # Over budget — apply tiered tool clearing first
        logger.info(
            f"[HISTORY] Token estimate {estimated_tokens} exceeds budget "
            f"{config.token_budget}, applying tiered tool clearing"
        )
        prepared = self._apply_tiered_tool_clearing(history, config)
        tokens_after_clearing = self._estimate_tokens(prepared)

        if tokens_after_clearing <= config.token_budget:
            logger.info(
                f"[HISTORY] Tool clearing reduced tokens from {estimated_tokens} "
                f"to {tokens_after_clearing}, within budget — no summarization needed"
            )
            return prepared, False

        # Still over budget — full compaction
        logger.info(
            f"[HISTORY] After tool clearing: {tokens_after_clearing} tokens, "
            f"still exceeds budget {config.token_budget}, triggering compaction"
        )
        compacted = await self._compact(prepared, session_context, config)

        return compacted, True

    def _apply_tiered_tool_clearing(
        self, history: list[ModelMessage], config: HistoryConfig | None = None
    ) -> list[ModelMessage]:
        """Replace older tool results with placeholders, keeping N most recent intact."""
        effective_config = config or self.config
        protect_count = effective_config.protect_recent_tool_results

        tool_result_indices: list[int] = []
        for i, msg in enumerate(history):
            if self._is_tool_result_message(msg):
                tool_result_indices.append(i)

        if not tool_result_indices or len(tool_result_indices) <= protect_count:
            return history

        if protect_count == 0:
            indices_to_clear = set(tool_result_indices)
        else:
            indices_to_clear = set(tool_result_indices[:-protect_count])

        result = []
        for i, msg in enumerate(history):
            if i in indices_to_clear:
                msg_copy = copy.deepcopy(msg)
                if isinstance(msg_copy, ModelRequest):
                    new_parts = []
                    for part in msg_copy.parts:
                        if isinstance(part, ToolReturnPart):
                            new_parts.append(
                                ToolReturnPart(
                                    tool_name=part.tool_name,
                                    content=TOOL_RESULT_PLACEHOLDER,
                                    tool_call_id=part.tool_call_id,
                                    timestamp=part.timestamp,
                                )
                            )
                        else:
                            new_parts.append(part)
                    msg_copy.parts = new_parts
                result.append(msg_copy)
            else:
                result.append(msg)

        return result

    @staticmethod
    def _resolve_dangling_tool_calls(
        history: list[ModelMessage],
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[ModelMessage]:
        """Add synthetic tool results for unresolved tool calls anywhere in history.

        Args:
            exclude_ids: Tool call IDs to skip resolution for (e.g., the pending
                host tool that DeferredToolResults will handle).
        """
        if not history:
            return history

        _exclude = exclude_ids or set()

        resolved: list[ModelMessage] = []
        pending_calls: dict[str, ToolCallPart] = {}
        inserted_synthetic = False
        inserted_count = 0

        for message in history:
            if pending_calls and not HistoryManager._is_pure_tool_result_message(message):
                inserted_count += len(pending_calls)
                resolved.append(
                    HistoryManager._make_synthetic_tool_result_message(pending_calls.values())
                )
                inserted_synthetic = True
                pending_calls = {}

            resolved.append(message)

            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart) and part.tool_call_id not in _exclude:
                        pending_calls[part.tool_call_id] = part
                continue

            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        pending_calls.pop(part.tool_call_id, None)

        if pending_calls:
            inserted_count += len(pending_calls)
            resolved.append(
                HistoryManager._make_synthetic_tool_result_message(pending_calls.values())
            )
            inserted_synthetic = True

        if inserted_synthetic:
            logger.warning(
                "[HISTORY] Added synthetic tool results for unresolved tool calls",
                count=inserted_count,
                excluded_ids=list(_exclude) if _exclude else None,
            )
            return resolved

        return history

    @staticmethod
    def _make_synthetic_tool_result_message(tool_calls: Iterable[ToolCallPart]) -> ModelRequest:
        """Create synthetic tool results for unresolved tool calls."""
        synthetic_parts = [
            ToolReturnPart(
                tool_name=tool_call.tool_name,
                content=(
                    "[Tool execution was interrupted by a system error on the previous turn. "
                    "The action did not complete. You may retry if needed.]"
                ),
                tool_call_id=tool_call.tool_call_id,
            )
            for tool_call in tool_calls
        ]
        return ModelRequest(parts=synthetic_parts)

    def _estimate_tokens(self, history: list[ModelMessage]) -> int:
        """Estimate token count using char_count / 4 heuristic."""
        total_chars = 0
        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            total_chars += len(content)
                    elif isinstance(part, ToolReturnPart):
                        content = part.content
                        if isinstance(content, str):
                            total_chars += len(content)
                        else:
                            total_chars += len(json.dumps(content, default=str))
                    elif isinstance(part, RetryPromptPart):
                        content = part.content
                        if isinstance(content, str):
                            total_chars += len(content)
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        total_chars += len(part.content)
                    elif isinstance(part, ToolCallPart):
                        args = part.args
                        if isinstance(args, str):
                            total_chars += len(args)
                        else:
                            total_chars += len(json.dumps(args, default=str))
                    elif isinstance(part, ThinkingPart) and part.content:
                        total_chars += len(part.content)
        return total_chars // 4

    async def _compact(
        self,
        history: list[ModelMessage],
        session_context: dict[str, Any],
        config: HistoryConfig | None = None,
    ) -> list[ModelMessage]:
        """Compact history with head preservation and structured summarization."""
        effective_config = config or self.config
        retain_count = effective_config.retain_recent
        if retain_count >= len(history):
            return history

        first_user_msg = self._extract_first_user_message(history)

        # Find a safe split point. The naive boundary (len - retain_count)
        # could land between a ModelResponse(ToolCallPart) and its paired
        # ModelRequest(ToolReturnPart), orphaning the tool result. Shift
        # the boundary forward past any pure tool-result messages.
        split = len(history) - retain_count
        while split < len(history) and self._is_pure_tool_result_message(history[split]):
            split += 1

        to_summarize: list[ModelMessage] = history[:split]
        to_keep: list[ModelMessage] = history[split:]

        existing_summary: str | None = None
        if to_summarize and self._is_summary_message(to_summarize[0]):
            existing_summary = self._extract_summary_content(to_summarize[0])
            to_summarize = to_summarize[1:]

        # Remove first user msg from to_summarize if it's the head-preserved one
        if (
            first_user_msg
            and to_summarize
            and isinstance(to_summarize[0], ModelRequest)
            and isinstance(first_user_msg, ModelRequest)
            and self._model_requests_match(to_summarize[0], first_user_msg)
        ):
            to_summarize = to_summarize[1:]

        if not to_summarize:
            logger.debug("[HISTORY] No messages to summarize after filtering")
            return history

        logger.info(
            f"[HISTORY] Compacting: summarizing {len(to_summarize)} messages, "
            f"retaining {len(to_keep)} recent messages"
        )

        summarizer_dicts = self._format_typed_to_dicts(
            to_summarize, truncation_limit=effective_config.message_truncation_limit
        )

        compaction_result = await self._summarizer.summarize_structured(
            summarizer_dicts,
            existing_summary,
        )

        wm_dict = compaction_result.working_memory.model_dump()
        session_context["working_memory"] = wm_dict

        summary_content = compaction_result.to_summary_message()
        summary_msg = ModelRequest(
            parts=[UserPromptPart(content=f"{SUMMARY_MARKER}\n{summary_content}")]
        )

        # Skip head preservation if first user message is in retained window
        first_already_retained = False
        if first_user_msg and to_keep:
            for msg in to_keep:
                if (
                    isinstance(msg, ModelRequest)
                    and isinstance(first_user_msg, ModelRequest)
                    and self._model_requests_match(msg, first_user_msg)
                ):
                    first_already_retained = True
                    break

        result: list[ModelMessage] = []
        if first_user_msg and not first_already_retained:
            result.append(first_user_msg)
        result.append(summary_msg)
        result.extend(to_keep)

        logger.info(f"[HISTORY] Compacted history from {len(history)} to {len(result)} messages")

        return result

    def _extract_first_user_message(self, history: list[ModelMessage]) -> ModelMessage | None:
        """Find the first user message, skipping summary messages."""
        for msg in history:
            if self._is_summary_message(msg):
                continue
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return copy.deepcopy(msg)
        return None

    @staticmethod
    def _is_summary_message(message: ModelMessage) -> bool:
        """Check if a message is a conversation summary."""
        if not isinstance(message, ModelRequest):
            return False
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                content = part.content
                if isinstance(content, str) and SUMMARY_MARKER in content:
                    return True
        return False

    def _extract_summary_content(self, message: ModelMessage) -> str:
        """Extract the summary text from a summary message."""
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    if isinstance(content, str) and SUMMARY_MARKER in content:
                        return content.replace(SUMMARY_MARKER, "").strip()
        logger.debug(f"[HISTORY] Could not extract summary content from message: {type(message)}")
        return ""

    @staticmethod
    def _is_tool_result_message(message: ModelMessage) -> bool:
        """Check if a message contains a tool result."""
        if not isinstance(message, ModelRequest):
            return False
        return any(isinstance(part, ToolReturnPart) for part in message.parts)

    @staticmethod
    def _is_pure_tool_result_message(message: ModelMessage) -> bool:
        """Check whether a request contains only tool return parts."""
        if not isinstance(message, ModelRequest) or not message.parts:
            return False
        return all(isinstance(part, ToolReturnPart) for part in message.parts)

    @staticmethod
    def _format_typed_to_dicts(
        messages: list[ModelMessage],
        truncation_limit: int = _DEFAULT_MESSAGE_TRUNCATION_LIMIT,
    ) -> list[dict[str, Any]]:
        """Convert typed messages to dict format for summarizer consumption."""

        def _truncate(text: str) -> str:
            if len(text) > truncation_limit:
                return text[:truncation_limit] + "... [truncated]"
            return text

        result = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        content = (
                            part.content if isinstance(part.content, str) else str(part.content)
                        )
                        result.append({"role": "user", "content": _truncate(content)})
                    elif isinstance(part, ToolReturnPart):
                        content = part.content
                        if not isinstance(content, str):
                            content = json.dumps(content, default=str)
                        full_content = f"Tool result ({part.tool_name}): {content}"
                        result.append({"role": "user", "content": _truncate(full_content)})
            elif isinstance(msg, ModelResponse):
                thinking_parts = []
                text_parts = []
                tool_parts = []
                for part in msg.parts:
                    if isinstance(part, ThinkingPart):
                        if part.content and part.content.strip():
                            truncated = _truncate(f"[Agent thinking]: {part.content}")
                            thinking_parts.append(truncated)
                    elif isinstance(part, TextPart):
                        text_parts.append(part.content)
                    elif isinstance(part, ToolCallPart):
                        args_str = (
                            part.args
                            if isinstance(part.args, str)
                            else json.dumps(part.args, default=str)
                        )
                        tool_parts.append(f"[Called {part.tool_name}({args_str})]")
                content_parts = thinking_parts + text_parts + tool_parts
                if content_parts:
                    combined = "\n".join(content_parts)
                    result.append({"role": "assistant", "content": _truncate(combined)})
        return result

    @staticmethod
    def _model_requests_match(a: ModelRequest, b: ModelRequest) -> bool:
        """Check if two ModelRequest messages match by comparing UserPromptPart content."""
        a_content = None
        b_content = None
        for part in a.parts:
            if isinstance(part, UserPromptPart):
                a_content = part.content
                break
        for part in b.parts:
            if isinstance(part, UserPromptPart):
                b_content = part.content
                break
        return a_content is not None and a_content == b_content
