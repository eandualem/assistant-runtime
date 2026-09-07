"""HistoryService — conversation history management with context window optimization.

Public facade for the history module. Implements LifecycleAware protocol.
The model-input policy is exposed as a native ``ProcessHistory`` capability
through :meth:`HistoryService.processor`, so it runs inside Pydantic AI's
request pipeline rather than ahead of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.usage import RunUsage

from assistant_runtime.services.history._manager import HistoryManager
from assistant_runtime.services.history._summarizer import HistorySummarizer
from assistant_runtime.services.history.config import HistoryConfig
from assistant_runtime.services.history.exceptions import CompactionError
from assistant_runtime.services.history.models import HistoryPreparationResult, WorkingMemory

if TYPE_CHECKING:
    from assistant_runtime.services.llm.interface import LlmService


class HistoryProcessor:
    """One turn's model-input policy, attachable as a native capability.

    Pydantic AI calls :meth:`process` before every model request of the
    run(s) it is attached to. ``result`` describes the most recent model
    input. The source messages are never modified, and the current run's
    own messages pass through verbatim so the run still reports them as new.
    """

    def __init__(self, manager: HistoryManager, session_context: dict[str, Any]) -> None:
        self._manager = manager
        self._session_context = session_context
        self.result: HistoryPreparationResult | None = None
        self.usage = RunUsage()
        """Usage of the summarisation calls this turn made, for the turn's accounting."""

    def capability(self) -> ProcessHistory:
        """The native capability wrapping this processor."""
        return ProcessHistory(self.process)

    async def process(
        self, ctx: RunContext[Any], messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        """Return the messages the model should see for this request."""
        frozen_from = next(
            (i for i, m in enumerate(messages) if getattr(m, "run_id", None) == ctx.run_id),
            len(messages),
        )
        try:
            prepared, was_compacted = await self._manager.prepare_history(
                messages, self._session_context, frozen_from=frozen_from, usage=self.usage
            )
        except Exception as e:
            if isinstance(e, CompactionError):
                raise
            raise CompactionError(f"History preparation failed: {e}") from e

        self.result = HistoryPreparationResult(
            was_compacted=was_compacted,
            message_count=len(prepared),
            estimated_tokens=self._manager._estimate_tokens(prepared),
            compacted_from=len(messages) if was_compacted else 0,
            message_summaries=HistoryService._summarize_messages(prepared),
        )
        return prepared


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

    def processor(self, session_context: dict[str, Any]) -> HistoryProcessor | None:
        """A per-turn processor, or None when compaction is disabled.

        ``session_context`` receives the compaction cache; attach the
        processor's capability to every agent run of the turn.

        Raises:
            CompactionError: If the service is not started.
        """
        if self._manager is None:
            raise CompactionError("History service not started")
        if not self._config.compaction_enabled:
            return None
        return HistoryProcessor(self._manager, session_context)

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
        *,
        usage: RunUsage | None = None,
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
            current_wm, recent_messages, turn_number, usage=usage
        )
