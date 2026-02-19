"""Request and response models for the assistant module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from lovely_assistant.services.tools.models import DeferredToolRequest


@dataclass(frozen=True)
class PromptResult:
    """Result of building a system prompt — content plus fragment metadata."""

    content: str
    fragments: list[dict[str, Any]] = field(default_factory=list)


class AssistantRequest(BaseModel):
    """Input for a single assistant interaction."""

    session_id: str
    message: str
    machine_state: dict[str, Any] | None = None
    # For continuations (frontend returning tool result):
    tool_call_id: str | None = None
    tool_result: Any = None


class AssistantResult(BaseModel):
    """Output from a single assistant interaction."""

    content: str | None = Field(
        default=None, description="Text response (None if deferred tool call)"
    )
    model: str = Field(description="Model used for this turn")
    deferred_tool_request: DeferredToolRequest | None = Field(
        default=None, description="Frontend tool call to execute"
    )
    session_id: str
    turn_number: int

    @property
    def is_tool_call(self) -> bool:
        """Whether this result is a deferred tool call (not text)."""
        return self.deferred_tool_request is not None
