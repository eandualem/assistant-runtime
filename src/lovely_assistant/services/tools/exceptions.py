"""Exception hierarchy for the tool service module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class ToolError(LovelyAssistantError):
    """Base exception for all tool module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "tools")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class ToolValidationError(ToolError):
    """Tool definition or call validation failed."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class ToolExecutionError(ToolError):
    """Tool execution failed at runtime."""
