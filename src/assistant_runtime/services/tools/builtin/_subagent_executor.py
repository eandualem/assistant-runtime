"""Subagent executor — creates and runs subagent instances."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from assistant_runtime.services.llm.interface import LlmService
    from assistant_runtime.services.tools.builtin.subagent import SubagentDefinition


async def execute_subagent(
    *,
    definition: SubagentDefinition,
    task: str,
    context: str | None = None,
    llm_service: LlmService,
    backend_toolsets: list,
    usage: RunUsage | None = None,
    model_override: str | None = None,
    thinking_budget_override: int | None = None,
    max_iterations_override: int | None = None,
) -> dict[str, Any]:
    """Execute a subagent with the given definition and task.

    Creates a Pydantic AI agent via LlmService.build_agent(), runs it with the
    provided backend toolsets (no subagent toolset — prevents recursion), and
    aggregates token usage into the parent's RunUsage.

    Returns a dict with either {"result": ..., "_metadata": ...} on success
    or {"success": false, "error": ..., "error_code": ...} on failure.
    """
    start_time = time.monotonic()

    # Resolve configuration
    model = model_override or definition.default_model
    thinking_budget = thinking_budget_override or definition.default_thinking_budget
    max_iterations = max_iterations_override or definition.max_iterations

    # Build system prompt with optional context
    system_prompt = definition.system_prompt
    if context:
        system_prompt = f"{system_prompt}\n\n## Additional Context\n\n{context}"

    try:
        # Resolve model through LlmService (validates and normalizes)
        resolved_model = llm_service.resolve_model(model)

        # Create subagent via the same factory as the main agent
        agent = llm_service.build_agent(
            model=resolved_model,
            system_prompt=system_prompt,
            toolsets=backend_toolsets if backend_toolsets else None,
            output_type=str,
            thinking_budget=thinking_budget,
        )

        # Run with usage aggregation and iteration limits
        usage_tracker = usage or RunUsage()
        result = await agent.run(
            task,
            usage=usage_tracker,
            usage_limits=UsageLimits(request_limit=max_iterations),
        )

        # Extract metadata from message history
        duration = time.monotonic() - start_time
        metadata = _extract_metadata(result.all_messages(), duration)

        logger.info(
            "Subagent completed",
            subagent_id=definition.id,
            model=resolved_model,
            duration_seconds=round(duration, 2),
            tool_calls=metadata["tool_calls_count"],
        )

        return {
            "result": result.output,
            "model": resolved_model,
            "subagent_id": definition.id,
            "_metadata": metadata,
        }

    except Exception as e:
        duration = time.monotonic() - start_time
        logger.error(
            "Subagent execution failed",
            subagent_id=definition.id,
            error_type=e.__class__.__name__,
            error=str(e),
            duration_seconds=round(duration, 2),
        )
        return {
            "success": False,
            "error": str(e),
            "error_code": "SUBAGENT_EXECUTION_ERROR",
            "subagent_id": definition.id,
        }


def _extract_metadata(messages: list, duration: float) -> dict[str, Any]:
    """Extract lightweight metadata from the subagent's message history."""
    tool_call_ids: set[str] = set()
    tools_used: set[str] = set()
    iterations = 0
    input_tokens = 0
    output_tokens = 0

    for msg in messages:
        # Count model responses as iterations
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_ids.add(part.tool_call_id)
                    tools_used.add(part.tool_name)
                elif isinstance(part, ToolReturnPart):
                    # Already counted via ToolCallPart
                    pass

        # Count model requests as iterations
        if hasattr(msg, "kind") and msg.kind == "response":
            iterations += 1

        # Extract usage if available
        if hasattr(msg, "usage") and msg.usage:
            input_tokens += getattr(msg.usage, "input_tokens", 0) or 0
            output_tokens += getattr(msg.usage, "output_tokens", 0) or 0

    return {
        "iterations": iterations,
        "tool_calls_count": len(tool_call_ids),
        "tools_used": sorted(tools_used),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_seconds": round(duration, 2),
    }
