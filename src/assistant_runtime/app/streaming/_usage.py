"""Token usage snapshots from pydantic-ai run results."""

from __future__ import annotations

from typing import Any

from loguru import logger


def _usage_value(usage: Any, *names: str) -> int:
    """Read a usage field while tolerating provider-specific attribute names."""
    for name in names:
        value = getattr(usage, name, None)
        if isinstance(value, int):
            return value
    return 0


def usage_dict(result_or_usage: Any) -> dict[str, int] | None:
    """Extract ``input/output/total_tokens`` from a run result or a ``RunUsage``.

    ``AgentRunResult`` exposes ``.usage`` as a property; a bare ``RunUsage``
    has no ``.usage`` attribute and is used as-is.
    """
    try:
        usage = getattr(result_or_usage, "usage", result_or_usage)
    except Exception as e:
        logger.debug("Failed to extract usage stats", error=str(e))
        return None
    if usage is None:
        return None

    input_tokens = _usage_value(usage, "input_tokens", "request_tokens")
    output_tokens = _usage_value(usage, "output_tokens", "response_tokens")
    total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def merge_usage(*usage_dicts: dict[str, int] | None) -> dict[str, int] | None:
    """Sum usage snapshots across several runs; None when none carried usage."""
    merged = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    saw_usage = False
    for usage in usage_dicts:
        if not usage:
            continue
        saw_usage = True
        for key in merged:
            value = usage.get(key)
            if isinstance(value, int):
                merged[key] += value
    return merged if saw_usage else None


def cache_counts(usage: Any) -> tuple[int, int]:
    """``(cache_read, cache_write)`` token counts from a pydantic-ai 2 ``RunUsage``."""
    read = getattr(usage, "cache_read_tokens", 0) or 0
    write = getattr(usage, "cache_write_tokens", 0) or 0
    return int(read), int(write)
