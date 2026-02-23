"""Subagent definitions, registry, and tool registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger
from pydantic_ai import RunContext

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition


@dataclass
class SubagentDefinition:
    """Definition of a subagent that the main assistant can spawn."""

    id: str
    name: str
    description: str
    system_prompt: str
    default_model: str | None = None
    default_thinking_budget: int | None = None
    max_iterations: int = 10


RESEARCHER_SYSTEM_PROMPT = """\
You are a Research Agent operating within the Lovely Universe — an AI agent \
orchestration ecosystem. Your job is to perform deep, focused research on a \
given topic and return a comprehensive synthesis.

## Your Capabilities

You have access to backend tools for inspecting the agent ecosystem:
- Agent session management (list, inspect, message agents)
- GitHub issue management (search, read, create issues)
- Agent plan inspection and management
- Notes management
- Schedule management

## Research Guidelines

1. **Be thorough.** Use multiple tools to gather information from different \
angles. Cross-reference findings.
2. **Synthesize, don't just list.** Your output should be a coherent analysis, \
not a raw dump of tool results.
3. **Cite sources.** When referencing specific agent states, issues, or plans, \
include identifiers (session names, issue numbers, etc.).
4. **Stay focused.** Address the specific research task given to you. Don't \
go on tangents.
5. **Acknowledge gaps.** If you can't find information on something, say so \
explicitly rather than guessing.
"""


SUBAGENT_REGISTRY: dict[str, SubagentDefinition] = {
    "researcher": SubagentDefinition(
        id="researcher",
        name="Research Agent",
        description=(
            "Deep research agent that can investigate topics across the agent "
            "ecosystem — agent states, GitHub issues, plans, and system status. "
            "Use for complex queries that require gathering and synthesizing "
            "information from multiple sources."
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


def register_subagent_tools(registry: ToolRegistry) -> None:
    """Register the run_subagent tool with the registry."""

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
        from lovely_assistant.services.tools._subagent_executor import execute_subagent

        # llm_service and backend_toolsets are injected by the ToolService
        # via the _handler_deps dict attached to this handler
        deps = getattr(run_subagent, "_handler_deps", None)
        if deps is None:
            return {
                "error": "Subagent system not initialized",
                "error_code": "SUBAGENT_NOT_INITIALIZED",
            }

        runtime_settings = deps.get("runtime_settings")
        model_override = runtime_settings.get("subagent_model", None) if runtime_settings else None
        thinking_override = (
            runtime_settings.get("subagent_thinking_budget", None) if runtime_settings else None
        )

        return await execute_subagent(
            definition=definition,
            task=task,
            context=context,
            llm_service=deps["llm_service"],
            backend_toolsets=deps["get_backend_toolsets"](),
            usage=ctx.usage,
            model_override=model_override,
            thinking_budget_override=thinking_override,
        )

    registry.register_backend_tool(
        ToolDefinition(
            name="run_subagent",
            description=(
                "Delegate a focused research or analysis task to a subagent. "
                "The subagent has access to backend tools (agent state, GitHub "
                "issues, plans, notes) but cannot spawn further subagents. "
                "Use this for complex queries requiring multi-step investigation "
                "and synthesis. Available subagents: "
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
