"""Factory for tool service lifecycle registration."""

from typing import Any

from loguru import logger
from pydantic_ai.usage import UsageLimits

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.tools.interface import ToolService
from assistant_runtime.services.tools.providers import build_providers


async def register_tools(
    app_state: Any, lifecycle: LifecycleManager, *, settings: AppSettings | None = None
) -> None:
    """Create ToolService, store on app_state, register with lifecycle."""
    settings = settings if settings is not None else AppSettings()
    media_service = getattr(app_state, "media_service", None)
    llm_service = getattr(app_state, "llm_service", None)
    mcp_service = getattr(app_state, "mcp_service", None)
    artifact_service = getattr(app_state, "artifact_service", None)
    service = ToolService(
        config=settings.tools,
        media_service=media_service,
        llm_service=llm_service,
        mcp_service=mcp_service,
        artifact_service=artifact_service,
        providers=build_providers(settings.providers),
        subagent_usage_limits=_budget_limits(settings.assistant.budget),
        subagent_defaults={
            "subagent_model": settings.assistant.subagent_model,
            "subagent_thinking_budget": settings.assistant.subagent_thinking_budget,
        },
    )
    app_state.tool_service = service
    await lifecycle.register("tool_service", service)
    logger.info("Tool module registered")


def _budget_limits(budget: Any) -> UsageLimits | None:
    """``ASSISTANT__BUDGET__*`` as native limits, so a subagent honours the same ceilings."""
    limits = UsageLimits(
        tool_calls_limit=budget.tool_calls,
        input_tokens_limit=budget.input_tokens,
        output_tokens_limit=budget.output_tokens,
        total_tokens_limit=budget.total_tokens,
        cost_limit=budget.cost_usd,
    )
    configured = (
        limits.has_token_limits()
        or limits.tool_calls_limit is not None
        or limits.cost_limit is not None
    )
    return limits if configured else None
