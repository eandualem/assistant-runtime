"""Exception hierarchy for the streaming module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class StreamingError(LovelyAssistantError):
    """Base exception for all streaming module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "streaming")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class StreamSetupError(StreamingError):
    """Failed to set up streaming (agent creation, prompt build, etc)."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        super().__init__(message, **kwargs)


class StreamExecutionError(StreamingError):
    """Error during active streaming (agent iteration failed)."""


class EventLimitError(StreamingError):
    """Safety limit on events per stream exceeded."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)
