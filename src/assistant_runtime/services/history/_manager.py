"""Model-input history policy: token budget, tool-result clearing, summarization.

Internal module — only accessed through HistoryService (interface.py).

The manager never edits the messages it receives; it returns the list the
model should see. Dangling tool calls and orphaned results are repaired by
Pydantic AI's own request pipeline, so nothing here re-implements that.

Two-stage compaction when the estimate exceeds the token budget:
1. Tiered tool result clearing — older tool results become placeholders,
   keeping their outcome and provider metadata.
2. Summarization — older messages are replaced by an LLM-written summary.
   The summary is cached in the session context so a multi-step turn pays
   for it once and later requests extend it incrementally.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
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
from pydantic_ai.usage import RunUsage

from assistant_runtime.services.history._summarizer import HistorySummarizer
from assistant_runtime.services.history.config import HistoryConfig

SUMMARY_MARKER = "[CONVERSATION SUMMARY]"
TOOL_RESULT_PLACEHOLDER = "[Tool result cleared to save context space]"
COMPACTION_CACHE_KEY = "history_compaction"
_DEFAULT_MESSAGE_TRUNCATION_LIMIT = 1000


class HistoryManager:
    """Decide what the model sees when the conversation exceeds the budget."""

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
        frozen_from: int | None = None,
        config_override: HistoryConfig | None = None,
        usage: RunUsage | None = None,
    ) -> tuple[list[ModelMessage], bool]:
        """Return the model input for ``history`` and whether it was summarized.

        Runs before every model request. Within budget the list is returned
        as is; otherwise tool clearing is applied, then summarization if the
        history is still over budget. Messages from ``frozen_from`` on are
        returned verbatim (the current run's own messages: Pydantic AI keeps
        the processed list as the run's history, so changing them would
        change what the run reports as new and what the runtime persists).
        """
        config = config_override or self.config

        if not history:
            return history, False
        frozen = len(history) if frozen_from is None else max(0, min(frozen_from, len(history)))

        estimated_tokens = self._estimate_tokens(history)

        if estimated_tokens <= config.token_budget:
            logger.debug(
                f"[HISTORY] Token estimate {estimated_tokens} within budget "
                f"{config.token_budget}, no compaction needed"
            )
            return history, False

        logger.info(
            f"[HISTORY] Token estimate {estimated_tokens} exceeds budget "
            f"{config.token_budget}, applying tiered tool clearing"
        )
        prepared = self._apply_tiered_tool_clearing(history, config, frozen_from=frozen)
        tokens_after_clearing = self._estimate_tokens(prepared)

        if tokens_after_clearing <= config.token_budget:
            logger.info(
                f"[HISTORY] Tool clearing reduced tokens from {estimated_tokens} "
                f"to {tokens_after_clearing}, within budget — no summarization needed"
            )
            return prepared, False

        logger.info(
            f"[HISTORY] After tool clearing: {tokens_after_clearing} tokens, "
            f"still exceeds budget {config.token_budget}, triggering compaction"
        )
        compacted = await self._compact(
            prepared, session_context, config, frozen_from=frozen, usage=usage
        )

        return compacted, compacted is not prepared

    def _apply_tiered_tool_clearing(
        self,
        history: list[ModelMessage],
        config: HistoryConfig | None = None,
        *,
        frozen_from: int | None = None,
    ) -> list[ModelMessage]:
        """Replace older tool results with placeholders, keeping the N most recent.

        Results are counted individually, so a response that called several
        tools keeps or loses each result on its own. Every other attribute of
        the part (outcome, metadata, timestamp, tool_call_id) is preserved.
        Results at or after ``frozen_from`` are never cleared.
        """
        effective_config = config or self.config
        protect_count = effective_config.protect_recent_tool_results
        frozen = len(history) if frozen_from is None else frozen_from

        result_ids = [
            part.tool_call_id
            for msg in history
            if isinstance(msg, ModelRequest)
            for part in msg.parts
            if isinstance(part, ToolReturnPart)
        ]
        if len(result_ids) <= protect_count:
            return history
        clear_ids = set(result_ids[: len(result_ids) - protect_count])

        result: list[ModelMessage] = []
        for index, msg in enumerate(history):
            if index >= frozen:
                result.append(msg)
                continue
            if isinstance(msg, ModelRequest) and any(
                isinstance(part, ToolReturnPart) and part.tool_call_id in clear_ids
                for part in msg.parts
            ):
                result.append(
                    replace(
                        msg,
                        parts=[
                            replace(part, content=TOOL_RESULT_PLACEHOLDER)
                            if isinstance(part, ToolReturnPart) and part.tool_call_id in clear_ids
                            else part
                            for part in msg.parts
                        ],
                    )
                )
            else:
                result.append(msg)

        return result

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
        *,
        frozen_from: int | None = None,
        usage: RunUsage | None = None,
    ) -> list[ModelMessage]:
        """Replace older messages with a summary, keeping the head and the recent tail.

        A summary cached in ``session_context`` for a prefix of this history
        is reused while the result fits the budget; when it no longer fits,
        the messages that have since left the retained window are folded
        into it. Returns ``history`` itself when nothing can be summarized.
        """
        effective_config = config or self.config
        retain_count = effective_config.retain_recent
        frozen = len(history) if frozen_from is None else frozen_from

        # Find a safe split point. The naive boundary (len - retain_count)
        # could land between a ModelResponse(ToolCallPart) and its paired
        # ModelRequest(ToolReturnPart), orphaning the tool result. Move the
        # boundary back so the call is retained with its result.
        split = min(len(history) - retain_count, frozen)
        while split > 0 and self._is_pure_tool_result_message(history[split]):
            split -= 1
        if split <= 0:
            return history

        first_user_msg = self._extract_first_user_message(history)

        existing_summary: str | None = None
        start = 0
        cached = self._cached_summary(history, session_context)
        if cached is not None:
            start, existing_summary = cached
            reused = self._assemble(first_user_msg, existing_summary, history[start:])
            if split <= start or self._estimate_tokens(reused) <= effective_config.token_budget:
                logger.debug("[HISTORY] Reusing cached summary", covered_messages=start)
                return reused

        to_summarize = history[start:split]
        # The head message is kept verbatim rather than summarized.
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
            if existing_summary is not None:
                return self._assemble(first_user_msg, existing_summary, history[start:])
            return history

        logger.info(
            f"[HISTORY] Compacting: summarizing {len(to_summarize)} messages, "
            f"retaining {len(history) - split} recent messages"
        )

        summarizer_dicts = self._format_typed_to_dicts(
            to_summarize, truncation_limit=effective_config.message_truncation_limit
        )
        compaction_result = await self._summarizer.summarize_structured(
            summarizer_dicts, existing_summary, usage=usage
        )
        summary_content = compaction_result.to_summary_message()
        session_context[COMPACTION_CACHE_KEY] = {
            "prefix_count": split,
            "fingerprint": self._fingerprint(history[:split]),
            "summary": summary_content,
        }

        result = self._assemble(first_user_msg, summary_content, history[split:])
        logger.info(f"[HISTORY] Compacted history from {len(history)} to {len(result)} messages")
        return result

    def _cached_summary(
        self, history: list[ModelMessage], session_context: dict[str, Any]
    ) -> tuple[int, str] | None:
        """``(prefix_count, summary)`` when the cache still describes a prefix of ``history``."""
        cached = session_context.get(COMPACTION_CACHE_KEY)
        if not isinstance(cached, dict):
            return None
        count = cached.get("prefix_count")
        summary = cached.get("summary")
        if not isinstance(count, int) or not isinstance(summary, str) or count > len(history):
            return None
        if self._fingerprint(history[:count]) != cached.get("fingerprint"):
            # A different branch is active; its summary does not apply.
            return None
        return count, summary

    @staticmethod
    def _fingerprint(messages: list[ModelMessage]) -> str:
        """Identity of a message prefix that survives tool-result clearing."""
        digest = hashlib.blake2b(digest_size=16)
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart | ToolCallPart):
                    key = f"{part.part_kind}:{part.tool_call_id}"
                else:
                    content = getattr(part, "content", "")
                    key = f"{part.part_kind}:{len(content) if isinstance(content, str) else 0}"
                digest.update(key.encode())
                digest.update(b"\0")
            digest.update(b"\n")
        return digest.hexdigest()

    def _assemble(
        self, first_user_msg: ModelMessage | None, summary: str, tail: list[ModelMessage]
    ) -> list[ModelMessage]:
        """Head message (unless already in the tail), the summary, then the tail."""
        result: list[ModelMessage] = []
        if first_user_msg is not None and not any(
            isinstance(msg, ModelRequest)
            and isinstance(first_user_msg, ModelRequest)
            and self._model_requests_match(msg, first_user_msg)
            for msg in tail
        ):
            result.append(first_user_msg)
        result.append(ModelRequest(parts=[UserPromptPart(content=f"{SUMMARY_MARKER}\n{summary}")]))
        result.extend(tail)
        return result

    def _extract_first_user_message(self, history: list[ModelMessage]) -> ModelMessage | None:
        """Find the first user message."""
        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        return copy.deepcopy(msg)
        return None

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
