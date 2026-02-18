"""Configuration for the tool service module."""

from pydantic import BaseModel, ConfigDict, Field


class ToolConfig(BaseModel):
    """Tool module configuration. Nested into AppSettings as `tools`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tools_per_request: int = Field(default=20, ge=1, le=100)
    tool_timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    enable_frontend_tools: bool = Field(default=True)
