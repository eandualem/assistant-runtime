"""Structured logging helpers — tag constants and payload truncation.

Lightweight module-level utilities for consistent log prefixes across the codebase.
Not a logging framework — just conventions and helpers.
"""

from __future__ import annotations

FULL_LOG_THRESHOLD = 500


def truncate(text: str, max_length: int = FULL_LOG_THRESHOLD) -> str:
    """Truncate text for log messages, preserving a readable prefix."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"... [{len(text) - max_length} chars truncated]"
