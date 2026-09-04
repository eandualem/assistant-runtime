"""HistoryService — conversation history management with context window optimization.

Public facade for the history module. Implements LifecycleAware protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse

from assistant_runtime.services.history._manager import HistoryManager
from assistant_runtime.services.history._summarizer import HistorySummarizer
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.exceptions import CompactionError
from assistant_runtime.services.history.models import HistoryPreparationResult, WorkingMemory

if TYPE_CHECKING:
    from assistant_runtime.services.llm.interface import LlmService


class HistoryService:
    """Conversation history management with context window optimization.

    Implements LifecycleAware. Delegates to internal HistoryManager and
    HistorySummarizer for the actual work.
    """

    def __init__(self, config: HistoryConfig, llm_service: LlmService) -> None:
        self._config = config
        self._llm = llm_service
        self._manager: HistoryManager | None = None
        self._runtime_settings = None
        self._started = False

    async def start(self) -> None:
        """Initialize internal components."""
        summarizer = HistorySummarizer(self._config, self._llm, self._runtime_settings)
        self._manager = HistoryManager(self._config, summarizer)
        self._started = True
        logger.info("History service started", token_budget=self._config.token_budget)

    async def stop(self) -> None:
        """Shutdown the history service."""
        self._manager = None
        self._started = False
        logger.info("History service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        return {
            "healthy": self._started,
            "token_budget": self._config.token_budget,
        }

    def set_runtime_settings(self, runtime_settings: Any | None) -> None:
        """Attach live runtime settings and propagate to internals."""
        self._runtime_settings = runtime_settings
        if self._manager is not None:
            self._manager.set_runtime_settings(runtime_settings)

    async def prepare_history_with_metadata(
        self,
        history: list[ModelMessage],
        session_context: dict[str, Any],
        *,
        is_continuation: bool = False,
        exclude_tool_call_ids: set[str] | None = None,
    ) -> HistoryPreparationResult:
        """Prepare history and return debug metadata.

        Args:
            history: Current conversation history.
            session_context: Session context dict.
            is_continuation: If True, passed through to manager.
            exclude_tool_call_ids: Tool call IDs to exclude from dangling resolution.

        Returns:
            HistoryPreparationResult with history and debug metadata.

        Raises:
            CompactionError: If history preparation fails.
        """
        if self._manager is None:
            raise CompactionError("History service not started")

        original_count = len(history)

        try:
            prepared, was_compacted = await self._manager.prepare_history(
                history,
                session_context,
                is_continuation=is_continuation,
                exclude_tool_call_ids=exclude_tool_call_ids,
            )
        except Exception as e:
            if isinstance(e, CompactionError):
                raise
            raise CompactionError(f"History preparation failed: {e}") from e

        estimated_tokens = self._manager._estimate_tokens(prepared)
        summaries = self._summarize_messages(prepared)

        return HistoryPreparationResult(
            history=prepared,
            was_compacted=was_compacted,
            message_count=len(prepared),
            estimated_tokens=estimated_tokens,
            compacted_from=original_count if was_compacted else 0,
            message_summaries=summaries,
        )

    @staticmethod
    def _summarize_messages(
        messages: list[ModelMessage],
        preview_limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Build per-message summaries for debug events."""
        summaries: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                content_parts = []
                for part in msg.parts:
                    text = getattr(part, "content", None)
                    if text is not None and not isinstance(text, str):
                        text = str(text)
                    if text:
                        content_parts.append(text)
                full = "\n".join(content_parts)
                preview = full[:preview_limit] if len(full) > preview_limit else full
                summaries.append(
                    {
                        "role": "user",
                        "content_preview": preview,
                        "char_count": len(full),
                        "part_count": len(msg.parts),
                    }
                )
            elif isinstance(msg, ModelResponse):
                content_parts = []
                for part in msg.parts:
                    text = getattr(part, "content", None)
                    if text is not None and not isinstance(text, str):
                        text = str(text)
                    if text:
                        content_parts.append(text)
                full = "\n".join(content_parts)
                preview = full[:preview_limit] if len(full) > preview_limit else full
                summaries.append(
                    {
                        "role": "assistant",
                        "content_preview": preview,
                        "char_count": len(full),
                        "part_count": len(msg.parts),
                    }
                )
            else:
                summaries.append(
                    {
                        "role": "system",
                        "content_preview": "",
                        "char_count": 0,
                        "part_count": 0,
                    }
                )
        return summaries

    async def extract_memory_delta(
        self,
        current_wm: WorkingMemory,
        recent_messages: list[dict[str, Any]],
        turn_number: int,
    ) -> WorkingMemory:
        """Extract working memory updates from recent messages.

        Args:
            current_wm: Current working memory state.
            recent_messages: Recent messages as dicts for analysis.
            turn_number: Current turn number.

        Returns:
            Updated WorkingMemory with deltas applied.
        """
        if self._manager is None:
            raise CompactionError("History service not started")

        return await self._manager._summarizer.extract_memory_delta(
            current_wm, recent_messages, turn_number
        )
