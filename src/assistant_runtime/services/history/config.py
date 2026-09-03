"""Configuration for the conversation history module."""

from pydantic import BaseModel, ConfigDict, Field


class HistoryConfig(BaseModel):
    """Configuration for conversation history management."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    token_budget: int = Field(default=100_000, ge=5000, le=500_000)
    retain_recent: int = Field(default=5, ge=1, le=50)
    protect_recent_tool_results: int = Field(default=3, ge=0, le=20)
    summarization_model: str | None = Field(default=None)
    working_memory_enabled: bool = Field(default=True)
    working_memory_model: str | None = Field(default=None)
    message_truncation_limit: int = Field(default=1000, ge=100, le=10000)
    max_memory_entries: int = Field(default=15, ge=5, le=50)
