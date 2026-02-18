"""HistoryService — conversation history management with context window optimization.

Public facade for the history module. Implements LifecycleAware protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.messages import ModelMessage

from lovely_assistant.services.history._manager import HistoryManager
from lovely_assistant.services.history._summarizer import HistorySummarizer
from lovely_assistant.services.history.config import HistoryConfig
from lovely_assistant.services.history.exceptions import CompactionError
from lovely_assistant.services.history.models import WorkingMemory

if TYPE_CHECKING:
    from lovely_assistant.services.llm.interface import LlmService


class HistoryService:
    """Conversation history management with context window optimization.

    Implements LifecycleAware. Delegates to internal HistoryManager and
    HistorySummarizer for the actual work.
    """

    def __init__(self, config: HistoryConfig, llm_service: LlmService) -> None:
        self._config = config
        self._llm = llm_service
        self._manager: HistoryManager | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize internal components."""
        summarizer = HistorySummarizer(self._config, self._llm)
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

    async def prepare_history(
        self,
        history: list[ModelMessage],
        session_context: dict[str, Any],
        *,
        is_continuation: bool = False,
    ) -> tuple[list[ModelMessage], bool]:
        """Prepare history for the agent loop.

        Args:
            history: Current conversation history.
            session_context: Session context dict (may be updated with working memory).
            is_continuation: If True, dangling tool calls are expected (deferred UI tool).

        Returns:
            Tuple of (prepared_history, context_was_modified).

        Raises:
            CompactionError: If history preparation fails.
        """
        if self._manager is None:
            raise CompactionError("History service not started")

        try:
            return await self._manager.prepare_history(
                history, session_context, is_continuation=is_continuation
            )
        except Exception as e:
            if isinstance(e, CompactionError):
                raise
            raise CompactionError(f"History preparation failed: {e}") from e

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
