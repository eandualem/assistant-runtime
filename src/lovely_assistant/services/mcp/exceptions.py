"""Exception hierarchy for the MCP service module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class MCPError(LovelyAssistantError):
    """Base exception for all MCP module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "mcp")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)
