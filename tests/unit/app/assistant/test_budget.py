"""Merging the host's budget settings with the definition's native limits."""

from __future__ import annotations

from pydantic_ai.usage import UsageLimits

from assistant_runtime.app.assistant._budget import build_usage_limits
from assistant_runtime.app.assistant.config import UsageBudget


def test_settings_alone():
    limits = build_usage_limits(
        UsageBudget(
            tool_calls=3, input_tokens=100, output_tokens=200, total_tokens=300, cost_usd=0.5
        ),
        None,
        request_limit=4,
    )
    assert limits == UsageLimits(
        request_limit=4,
        tool_calls_limit=3,
        input_tokens_limit=100,
        output_tokens_limit=200,
        total_tokens_limit=300,
        cost_limit=0.5,
    )


def test_nothing_set_leaves_only_the_request_limit():
    limits = build_usage_limits(UsageBudget(), None, request_limit=10)
    assert limits == UsageLimits(request_limit=10)


def test_stricter_of_settings_and_definition_wins():
    native = UsageLimits(
        request_limit=2,
        tool_calls_limit=9,
        output_tokens_limit=50,
        cost_limit=1.0,
        per_request_input_tokens_limit=4000,
        count_tokens_before_request=True,
    )
    limits = build_usage_limits(
        UsageBudget(tool_calls=3, output_tokens=200, cost_usd=0.25), native, request_limit=10
    )
    assert limits.request_limit == 2
    assert limits.tool_calls_limit == 3
    assert limits.output_tokens_limit == 50
    assert limits.cost_limit == 0.25
    assert limits.input_tokens_limit is None
    # Native-only knobs pass through untouched.
    assert limits.per_request_input_tokens_limit == 4000
    assert limits.count_tokens_before_request is True
