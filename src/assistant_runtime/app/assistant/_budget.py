"""Per-turn usage ceilings: settings, the definition's native limits, and the request."""

from __future__ import annotations

from pydantic_ai.usage import UsageLimits

from assistant_runtime.app.assistant.config import UsageBudget


def _strictest(*values: int | float | None) -> int | float | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def build_usage_limits(
    budget: UsageBudget, definition_limits: UsageLimits | None, *, request_limit: int | None
) -> UsageLimits:
    """The native limits for one turn.

    ``ASSISTANT__BUDGET__*`` and ``AssistantDefinition.usage_limits`` are both
    host ceilings; where both set a limit the stricter one applies.
    ``request_limit`` is the resolved ``max_turns`` (already clamped to what
    the host allows).
    """
    native = definition_limits or UsageLimits()
    return UsageLimits(
        request_limit=_strictest(request_limit, native.request_limit),
        tool_calls_limit=_strictest(budget.tool_calls, native.tool_calls_limit),
        input_tokens_limit=_strictest(budget.input_tokens, native.input_tokens_limit),
        output_tokens_limit=_strictest(budget.output_tokens, native.output_tokens_limit),
        total_tokens_limit=_strictest(budget.total_tokens, native.total_tokens_limit),
        cost_limit=_strictest(budget.cost_usd, native.cost_limit),
        per_request_input_tokens_limit=native.per_request_input_tokens_limit,
        count_tokens_before_request=native.count_tokens_before_request,
    )
