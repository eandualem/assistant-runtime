"""Data models for the tool service module."""

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ToolCategory(StrEnum):
    """Classification of tool execution location."""

    BACKEND = "backend"
    FRONTEND = "frontend"


class ToolDefinition(BaseModel):
    """Definition of a single tool — metadata and schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters_schema: dict[str, Any]
    category: ToolCategory
    timeout: float | None = None


class ToolSet(BaseModel):
    """Collection of tools available for a request."""

    backend_tools: list[ToolDefinition] = []
    frontend_tools: list[ToolDefinition] = []
    page: str | None = None
    filtered_out_count: int = 0

    @property
    def total_count(self) -> int:
        """Total number of tools in this set."""
        return len(self.backend_tools) + len(self.frontend_tools)

    @property
    def tool_names(self) -> list[str]:
        """All tool names in this set."""
        return [t.name for t in self.backend_tools] + [t.name for t in self.frontend_tools]


class DeferredToolRequest(BaseModel):
    """A tool call that must be forwarded to the frontend via SSE."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    arguments: dict[str, Any] = {}


class ToolResult(BaseModel):
    """Result of a tool execution (backend or deferred frontend)."""

    tool_name: str
    request_id: str
    content: Any
    is_error: bool = False
