"""Exception hierarchy for the assistant module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class AssistantError(AssistantRuntimeError):
    """Base exception for all assistant module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "assistant")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class PromptBuildError(AssistantError):
    """Failed to build the system prompt."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        super().__init__(message, **kwargs)


class AgentRunError(AssistantError):
    """Agent execution failed."""


class SessionError(AssistantError):
    """Session state management error."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        super().__init__(message, **kwargs)
