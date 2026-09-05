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
    snapshot: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": _usage_value(usage, "requests"),
        "tool_calls": _usage_value(usage, "tool_calls"),
        # None, not 0: a provider or model without a known price has no cost figure.
        "cost_usd": _cost(usage),
    }
    return snapshot


def _cost(usage: Any) -> float | None:
    cost = getattr(usage, "cost", None)
    if cost is None or callable(cost):
        return None
    try:
        return float(cost)
    except (TypeError, ValueError):
        return None


COUNTED_KEYS = ("input_tokens", "output_tokens", "total_tokens", "requests", "tool_calls")


def merge_usage(*usage_dicts: dict[str, Any] | None) -> dict[str, Any] | None:
    """Sum usage snapshots across several runs; None when none carried usage.

    Counts add up; ``cost_usd`` adds up only while every part reports one,
    otherwise it stays None. ``auxiliary`` sections are merged by name.
    """
    merged: dict[str, Any] = dict.fromkeys(COUNTED_KEYS, 0)
    cost: float | None = 0.0
    auxiliary: dict[str, Any] = {}
    saw_usage = False
    for usage in usage_dicts:
        if not isinstance(usage, dict) or not usage:
            continue
        saw_usage = True
        for key in COUNTED_KEYS:
            value = usage.get(key)
            if isinstance(value, int):
                merged[key] += value
        part_cost = usage.get("cost_usd")
        cost = cost + part_cost if cost is not None and isinstance(part_cost, int | float) else None
        for name, part in (usage.get("auxiliary") or {}).items():
            auxiliary[name] = merge_usage(auxiliary.get(name), part)
    if not saw_usage:
        return None
    merged["cost_usd"] = cost
    if auxiliary:
        merged["auxiliary"] = auxiliary
    return merged


def with_auxiliary(
    usage: dict[str, Any] | None, name: str, part: dict[str, Any] | None
) -> dict[str, Any] | None:
    """``usage`` with ``part`` recorded under ``auxiliary[name]`` (summed if present)."""
    if not isinstance(part, dict):
        return usage
    base = dict(usage if isinstance(usage, dict) else dict.fromkeys(COUNTED_KEYS, 0))
    auxiliary = dict(base.get("auxiliary") or {})
    auxiliary[name] = merge_usage(auxiliary.get(name), part)
    base["auxiliary"] = auxiliary
    base.setdefault("cost_usd", None)
    return base


def cache_counts(usage: Any) -> tuple[int, int]:
    """``(cache_read, cache_write)`` token counts from a pydantic-ai 2 ``RunUsage``."""
    read = getattr(usage, "cache_read_tokens", 0) or 0
    write = getattr(usage, "cache_write_tokens", 0) or 0
    return int(read), int(write)
