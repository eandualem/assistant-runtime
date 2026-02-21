"""Structured logging helpers — tag constants and payload truncation.

Lightweight module-level utilities for consistent log prefixes across the codebase.
Not a logging framework — just conventions and helpers.
"""

from __future__ import annotations

# --- Tag constants (prefix log messages for grep-ability) ---

TAG_REQUEST = "[REQUEST]"
TAG_STREAM = "[STREAM]"
TAG_TOOLS = "[TOOLS]"
TAG_SESSION = "[SESSION]"
TAG_DB = "[DB]"
TAG_LLM = "[LLM]"
TAG_HISTORY = "[HISTORY]"

# --- Truncation ---

FULL_LOG_THRESHOLD = 500
PREVIEW_LENGTH = 200


def truncate(text: str, max_length: int = FULL_LOG_THRESHOLD) -> str:
    """Truncate text for log messages, preserving a readable prefix."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} chars truncated]"


def truncate_payload(payload: object, max_length: int = FULL_LOG_THRESHOLD) -> str:
    """Truncate an arbitrary payload's string representation for logging."""
    text = str(payload)
    return truncate(text, max_length)
