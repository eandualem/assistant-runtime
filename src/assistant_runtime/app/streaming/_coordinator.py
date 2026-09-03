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

from assistant_runtime.app.streaming._event_builder import (
    make_agent_status_event,
    make_debug_thinking_event,
    make_final_response_event,
    make_text_delta_event,
    make_thinking_delta_event,
)
from assistant_runtime.app.streaming.exceptions import EventLimitError


class EventCoordinator:
    """Tracks emitted events for a single streaming request.

    Ensures protocol-level constraints are met and suppresses duplicates.
    """

    def __init__(self, max_events: int) -> None:
        self._max_events = max_events
        self._event_count = 0
        self._debug_event_count = 0
        self._started_emitted = False
        self._completed_emitted = False
        self._final_response_emitted = False
        self._debug_events: list[dict[str, Any]] = []
        self._streamed_text = False
        self._streamed_thinking = False
        self._thinking_buffer: list[str] = []
        self._response_buffer: list[str] = []
        self._current_segment_kind: str | None = None
        self._current_segment_id: str | None = None
        self._current_segment_index: int | None = None
        self._current_segment_delta_index = 0
        self._next_segment_index = 0

    @property
    def event_count(self) -> int:
        """Number of protocol events emitted so far."""
        return self._event_count

    @property
    def debug_event_count(self) -> int:
        """Number of debug events emitted so far."""
        return self._debug_event_count

    @property
    def has_streamed_text(self) -> bool:
        """Whether any text_delta events were emitted via emit_text_delta."""
        return self._streamed_text

    @property
    def has_streamed_thinking(self) -> bool:
        """Whether any thinking_delta events were emitted via emit_thinking_delta."""
        return self._streamed_thinking

    def try_started(self) -> dict[str, Any] | None:
        """Emit agent_status(started) if not already emitted."""
        if self._started_emitted:
            logger.warning("Duplicate agent_status(started) suppressed")
            return None
        self._started_emitted = True
        return self._track(make_agent_status_event("started"))

    def try_completed(self) -> dict[str, Any] | None:
        """Emit agent_status(completed) if not already emitted.

        Terminal event — bypasses event limit. The frontend MUST receive
        this to exit the "thinking" state.
        """
        if self._completed_emitted:
            logger.warning("Duplicate agent_status(completed) suppressed")
            return None
        self._completed_emitted = True
        self.clear_content_segment()
        event = make_agent_status_event("completed")
        self._event_count += 1  # count but never raise
        return event

    def try_final_response(
        self,
        content: str | None,
        model: str,
        *,
        session_id: str | None = None,
        message_id: str | None = None,
        trace_id: str | None = None,
        error: bool = False,
        error_type: str | None = None,
        usage: dict[str, int] | None = None,
        pending_tool_call: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Emit final_response if not already emitted.

        Terminal event — bypasses event limit. The frontend MUST receive
        this before completed.
        """
        if self._final_response_emitted:
            logger.warning("Duplicate final_response suppressed")
            return None
        self._final_response_emitted = True
        self.clear_content_segment()
        # Clear content when text was already delivered via deltas (unless error)
        effective_content = (
            "" if self._streamed_text and not error and content is not None else content
        )
        event = make_final_response_event(
            effective_content,
            model,
            session_id=session_id,
            message_id=message_id,
            trace_id=trace_id,
            streamed=self._streamed_text,
            thinking_streamed=self._streamed_thinking,
            error=error,
            error_type=error_type,
            usage=usage,
            pending_tool_call=pending_tool_call,
        )
        self._event_count += 1  # count but never raise
        return event

    def emit_text_delta(self, content: str) -> dict[str, Any]:
        """Create, track, and return a text_delta event. Sets streamed_text flag."""
        self._streamed_text = True
        self._response_buffer.append(content)
        segment = self._next_segment("text")
        return self._track(
            make_text_delta_event(
                content,
                segment_id=segment["segment_id"],
                segment_index=segment["segment_index"],
                delta_index=segment["delta_index"],
                segment_started=segment["segment_started"],
                segment_kind="text",
            )
        )

    def emit_thinking_delta(self, content: str) -> dict[str, Any]:
        """Create, track, and return a thinking_delta event. Sets streamed_thinking flag."""
        self._streamed_thinking = True
        self._thinking_buffer.append(content)
        segment = self._next_segment("thinking")
        return self._track(
            make_thinking_delta_event(
                content,
                segment_id=segment["segment_id"],
                segment_index=segment["segment_index"],
                delta_index=segment["delta_index"],
                segment_started=segment["segment_started"],
                segment_kind="thinking",
            )
        )

    def flush_thinking(self) -> dict[str, Any] | None:
        """Flush current thinking buffer as a debug_thinking trace event.

        Returns the event if buffer had content, None otherwise.
        Clears the buffer so each flush captures only one iteration's thinking.
        """
        if not self._thinking_buffer:
            return None
        content = "".join(self._thinking_buffer)
        self._thinking_buffer.clear()
        return self.track_debug(make_debug_thinking_event(content))

    def track(self, event: dict[str, Any]) -> dict[str, Any]:
        """Track a regular event (thinking_delta, text_delta, tool_*, error)."""
        if event.get("type") not in {"thinking_delta", "text_delta"}:
            self.clear_content_segment()
        return self._track(event)

    def track_debug(self, event: dict[str, Any]) -> dict[str, Any]:
        """Track a debug event — counted separately, never raises EventLimitError."""
        self._debug_event_count += 1
        self._debug_events.append(event)
        return event

    @property
    def debug_events(self) -> list[dict[str, Any]]:
        """All collected debug events for trace persistence."""
        return self._debug_events

    @property
    def accumulated_thinking(self) -> str:
        """Full thinking content accumulated from all thinking_delta events."""
        return "".join(self._thinking_buffer)

    @property
    def accumulated_response(self) -> str:
        """Full response content accumulated from all text_delta events."""
        return "".join(self._response_buffer)

    def clear_content_segment(self) -> None:
        """Reset active content-segment tracking on any non-content boundary."""
        self._current_segment_kind = None
        self._current_segment_id = None
        self._current_segment_index = None
        self._current_segment_delta_index = 0

    def _track(self, event: dict[str, Any]) -> dict[str, Any]:
        """Increment counter and enforce limit."""
        self._event_count += 1
        if self._event_count > self._max_events:
            raise EventLimitError(f"Event limit exceeded: {self._event_count} > {self._max_events}")
        return event

    def _next_segment(self, kind: str) -> dict[str, int | str | bool]:
        """Return metadata for the current segment, rotating on kind changes."""
        if self._current_segment_kind != kind:
            self.clear_content_segment()
            self._current_segment_kind = kind
            self._current_segment_index = self._next_segment_index
            self._current_segment_id = f"segment_{self._next_segment_index}"
            self._next_segment_index += 1

        segment_started = self._current_segment_delta_index == 0
        delta_index = self._current_segment_delta_index
        self._current_segment_delta_index += 1
        assert self._current_segment_id is not None
        assert self._current_segment_index is not None
        return {
            "segment_id": self._current_segment_id,
            "segment_index": self._current_segment_index,
            "delta_index": delta_index,
            "segment_started": segment_started,
        }
