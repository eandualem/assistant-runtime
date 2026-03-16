"""Request-scoped assistant tool context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_assistant_session_id: ContextVar[str | None] = ContextVar(
    "_current_assistant_session_id",
    default=None,
)


@contextmanager
def assistant_request_context(session_id: str) -> Iterator[None]:
    """Bind the current assistant session id for backend tool handlers."""
    token = _current_assistant_session_id.set(session_id)
    try:
        yield
    finally:
        _current_assistant_session_id.reset(token)


def get_current_assistant_session_id() -> str | None:
    """Return the active assistant session id for the current request."""
    return _current_assistant_session_id.get()
