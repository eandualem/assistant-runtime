"""Approvals: plans a peer agent proposes before acting, which the assistant can approve or reject.

A provider implements ``ApprovalsProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class ApprovalsProvider(Protocol):
    """What a provider of the approvals capability implements."""

    async def list_agent_plans(self) -> dict[str, Any]: ...

    async def approve_plan(self, session_name: str) -> dict[str, Any]: ...

    async def reject_plan(self, session_name: str, reason: str) -> dict[str, Any]: ...


def register_approvals_tools(registry: ToolRegistry, provider: ApprovalsProvider) -> None:
    """Register all plan management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="list_agent_plans",
            description=(
                "List all agents that have plans awaiting approval. "
                "Shows session name, plan title, plan file path, and how long "
                "the plan has been waiting."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        provider.list_agent_plans,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="approve_plan",
            description=(
                "Approve a pending plan for an agent. Sends the Shift+Tab key "
                "sequence to the agent's tmux session to trigger plan approval."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session with a pending plan",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.approve_plan,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="reject_plan",
            description=(
                "Reject a pending plan for an agent. Sends the rejection reason "
                "as text to the agent's tmux session."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session with a pending plan",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for rejecting the plan",
                    },
                },
                "required": ["session_name", "reason"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.reject_plan,
    )

    logger.info("Registered plan management tools", count=3)
