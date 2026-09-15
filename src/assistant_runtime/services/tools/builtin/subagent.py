"""Subagent definitions, registry, and tool registration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic_ai import RunContext
from pydantic_ai.usage import UsageLimits

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


@dataclass
class SubagentDefinition:
    """Definition of a subagent that the main assistant can spawn."""

    id: str
    name: str
    description: str
    system_prompt: str
    max_iterations: int = 10


RESEARCHER_SYSTEM_PROMPT = """\
You are a research agent working for an application's assistant. You are given \
one focused task; perform the multi-step investigation it needs and return a \
synthesis the assistant can act on.

## Your capabilities

You have the backend tools of the session that delegated to you. Use them to \
gather information; you cannot delegate further.

## Research guidelines

1. **Be thorough.** Use several tools and angles; cross-check what you find.
2. **Synthesize, don't just list.** Return a coherent analysis, not a dump of \
tool results.
3. **Cite sources.** Name the tool results, files, records or identifiers a \
finding rests on.
4. **Stay focused.** Answer the task you were given; note tangents, do not \
pursue them.
5. **Acknowledge gaps.** Say what you could not find rather than guessing.
"""


SUBAGENT_REGISTRY: dict[str, SubagentDefinition] = {
    "researcher": SubagentDefinition(
        id="researcher",
        name="Research Agent",
        description=(
            "Deep research agent for questions that need several tool calls and a "
            "synthesis of what they return."
        ),
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
    ),
}


def get_subagent(subagent_id: str) -> SubagentDefinition | None:
    """Look up a subagent definition by ID."""
    return SUBAGENT_REGISTRY.get(subagent_id)


def list_subagents() -> list[dict[str, Any]]:
    """Return summary info for all registered subagents."""
    return [
        {"id": d.id, "name": d.name, "description": d.description}
        for d in SUBAGENT_REGISTRY.values()
    ]


def register_subagent_tools(
    registry: ToolRegistry,
    llm_service: Any,
    *,
    backend_toolsets: Callable[[], list[Any]],
    runtime_settings: Callable[[], Any | None],
    usage_limits: UsageLimits | None = None,
) -> None:
    """Register ``run_subagent``.

    ``backend_toolsets`` and ``runtime_settings`` are callables because both
    change after registration: tools keep being registered, and the runtime
    settings overlay is attached once the lifecycle has started.
    ``usage_limits`` are the host's per-turn ceilings; a subagent run stays
    within them on top of its own iteration limit.
    """

    async def run_subagent(
        ctx: RunContext[None],
        task: str,
        subagent_id: str = "researcher",
        context: str | None = None,
    ) -> dict[str, Any]:
        """Delegate a focused task to a subagent.

        Spawns a subagent (e.g. researcher) to perform deep, multi-step work
        and returns its synthesized result. The subagent has access to backend
        tools but cannot spawn further subagents.

        Args:
            ctx: Pydantic AI run context (injected automatically).
            task: The task or question for the subagent to work on.
            subagent_id: Which subagent to use. Default: "researcher".
            context: Optional additional context to include in the subagent's prompt.
        """
        definition = get_subagent(subagent_id)
        if definition is None:
            available = [s["id"] for s in list_subagents()]
            return {
                "error": f"Unknown subagent '{subagent_id}'",
                "error_code": "SUBAGENT_NOT_FOUND",
                "available_subagents": available,
            }

        # Import here to avoid circular imports at module level
        from assistant_runtime.services.tools.builtin._subagent_executor import execute_subagent

        settings = runtime_settings()
        model_override = settings.get("subagent_model", None) if settings else None
        thinking_override = settings.get("subagent_thinking_budget", None) if settings else None

        return await execute_subagent(
            definition=definition,
            task=task,
            context=context,
            llm_service=llm_service,
            backend_toolsets=backend_toolsets(),
            usage=ctx.usage,
            model_override=model_override,
            thinking_budget_override=thinking_override,
            usage_limits=usage_limits,
        )

    registry.register_backend_tool(
        ToolDefinition(
            name="run_subagent",
            description=(
                "Delegate a focused research or analysis task to a subagent. "
                "The subagent has this session's backend tools but cannot spawn "
                "further subagents. Use it for questions that need multi-step "
                "investigation and synthesis. Available subagents: "
                + ", ".join(f"'{d.id}' ({d.description})" for d in SUBAGENT_REGISTRY.values())
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question for the subagent to work on",
                    },
                    "subagent_id": {
                        "type": "string",
                        "description": "Which subagent to use (default: 'researcher')",
                        "default": "researcher",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional additional context for the subagent",
                    },
                },
                "required": ["task"],
            },
            category=ToolCategory.BACKEND,
        ),
        run_subagent,
    )

    logger.info("Registered subagent tools", count=1)
