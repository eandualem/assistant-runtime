"""EventCoordinator — per-request event tracking and deduplication.

Enforces protocol constraints:
- Exactly one agent_status(started) per stream
- Exactly one final_response per stream
- Exactly one agent_status(completed) per stream
- Event count safety limit
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from lovely_assistant.app.streaming._event_builder import (
    make_agent_status_event,
    make_final_response_event,
)
from lovely_assistant.app.streaming.exceptions import EventLimitError


class EventCoordinator:
    """Tracks emitted events for a single streaming request.

    Ensures protocol-level constraints are met and suppresses duplicates.
    """

    def __init__(self, max_events: int) -> None:
        self._max_events = max_events
        self._event_count = 0
        self._started_emitted = False
        self._completed_emitted = False
        self._final_response_emitted = False

    @property
    def event_count(self) -> int:
        """Number of events emitted so far."""
        return self._event_count

    def try_started(self) -> dict[str, Any] | None:
        """Emit agent_status(started) if not already emitted."""
        if self._started_emitted:
            logger.warning("Duplicate agent_status(started) suppressed")
            return None
        self._started_emitted = True
        return self._track(make_agent_status_event("started"))

    def try_completed(self) -> dict[str, Any] | None:
        """Emit agent_status(completed) if not already emitted."""
        if self._completed_emitted:
            logger.warning("Duplicate agent_status(completed) suppressed")
            return None
        self._completed_emitted = True
        return self._track(make_agent_status_event("completed"))

    def try_final_response(self, content: str | None, model: str) -> dict[str, Any] | None:
        """Emit final_response if not already emitted."""
        if self._final_response_emitted:
            logger.warning("Duplicate final_response suppressed")
            return None
        self._final_response_emitted = True
        return self._track(make_final_response_event(content, model))

    def track(self, event: dict[str, Any]) -> dict[str, Any]:
        """Track a regular event (thinking_delta, text_delta, tool_*, error)."""
        return self._track(event)

    def _track(self, event: dict[str, Any]) -> dict[str, Any]:
        """Increment counter and enforce limit."""
        self._event_count += 1
        if self._event_count > self._max_events:
            raise EventLimitError(f"Event limit exceeded: {self._event_count} > {self._max_events}")
        return event
