"""Exception hierarchy for the assistant module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class AssistantError(LovelyAssistantError):
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


class ContinuationMismatchError(SessionError):
    """Continuation tool_call_id does not match the stored pending ID."""

    def __init__(self, expected: str, received: str, tool_name: str) -> None:
        self.expected = expected
        self.received = received
        self.tool_name = tool_name
        super().__init__(
            f"Continuation tool_call_id mismatch for '{tool_name}': "
            f"expected '{expected}', received '{received}'"
        )


REJECTED_TOOL_RESULT: dict[str, str] = {
    "error": "Only 1 frontend tool call is allowed per turn. This call was rejected.",
    "error_code": "MULTIPLE_FRONTEND_TOOLS",
    "suggestion": "Call frontend tools one at a time, waiting for each result.",
}
