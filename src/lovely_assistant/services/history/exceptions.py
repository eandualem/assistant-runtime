"""Exception hierarchy for the history service module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class HistoryError(LovelyAssistantError):
    """Base exception for all history module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "history")
        kwargs.setdefault("severity", "medium")
        super().__init__(message, **kwargs)


class CompactionError(HistoryError):
    """History compaction failed."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class SummarizationError(HistoryError):
    """LLM summarization failed."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)
