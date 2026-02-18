"""Internal in-memory session store.

Manages per-session context (working memory, turn count, pending tool calls).
Persistent storage is a later concern — this is an in-memory dict for now.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage


class SessionStore:
    """In-memory session state storage."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get or create session context."""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "turn_number": 0,
                "working_memory": None,
                "pending_tool_call": None,
                "message_history": [],
            }
        return self._sessions[session_id]

    def get_history(self, session_id: str) -> list[ModelMessage]:
        """Get message history for a session."""
        ctx = self.get_context(session_id)
        return ctx.get("message_history", [])

    def save_history(self, session_id: str, messages: list[ModelMessage]) -> None:
        """Save message history for a session."""
        ctx = self.get_context(session_id)
        ctx["message_history"] = messages

    def increment_turn(self, session_id: str) -> int:
        """Increment and return the turn number."""
        ctx = self.get_context(session_id)
        ctx["turn_number"] = ctx.get("turn_number", 0) + 1
        return ctx["turn_number"]

    def set_pending_tool_call(
        self,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
    ) -> None:
        """Store a pending frontend tool call for continuation."""
        ctx = self.get_context(session_id)
        ctx["pending_tool_call"] = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
        }

    def clear_pending_tool_call(self, session_id: str) -> dict[str, str] | None:
        """Clear and return the pending tool call, if any."""
        ctx = self.get_context(session_id)
        pending = ctx.get("pending_tool_call")
        ctx["pending_tool_call"] = None
        return pending

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        return session_id in self._sessions

    def session_count(self) -> int:
        """Number of active sessions."""
        return len(self._sessions)
