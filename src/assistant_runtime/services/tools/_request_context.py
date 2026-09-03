"""Request-scoped assistant tool context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_current_assistant_session_id: ContextVar[str | None] = ContextVar(
    "_current_assistant_session_id",
    default=None,
)
_current_telegram_chat_binding: ContextVar[str | None] = ContextVar(
    "_current_telegram_chat_binding",
    default=None,
)


@contextmanager
def assistant_request_context(session_id: str) -> Iterator[None]:
    """Bind the current assistant session id for backend tool handlers."""
    session_token = _current_assistant_session_id.set(session_id)
    telegram_token = _current_telegram_chat_binding.set(None)
    try:
        yield
    finally:
        _current_telegram_chat_binding.reset(telegram_token)
        _current_assistant_session_id.reset(session_token)


def get_current_assistant_session_id() -> str | None:
    """Return the active assistant session id for the current request."""
    return _current_assistant_session_id.get()


def record_current_telegram_chat_binding(chat_id: str) -> None:
    """Record the Telegram chat id used during the current assistant request."""
    _current_telegram_chat_binding.set(chat_id)


def get_current_telegram_chat_binding() -> str | None:
    """Return the Telegram chat id used during the current assistant request."""
    return _current_telegram_chat_binding.get()
