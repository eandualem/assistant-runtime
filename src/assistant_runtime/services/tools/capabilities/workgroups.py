"""Workgroups: several agents coordinated on one piece of work.

Create a group, follow it, report a worker's status, broadcast to it, close it.

A provider implements ``WorkgroupsProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

# The vocabulary of the capability; a provider maps these onto its own.
_SWARM_PHASES: frozenset[str] = frozenset(
    {
        "created",
        "planning",
        "working",
        "validating",
        "pr_open",
        "awaiting_review",
        "merged",
        "cleaned_up",
        "failed",
        "discarded",
    }
)
_SWARM_WORKER_ROLES: frozenset[str] = frozenset({"lead", "coder", "tester", "validator", "scout"})
_SWARM_WORKER_STATUSES: frozenset[str] = frozenset(
    {"pending", "started", "working", "pr_created", "done", "failed"}
)
_WORKER_REQUIRED_FIELDS: tuple[str, ...] = ("name", "role", "branch", "worktree_path", "session")


class WorkgroupsProvider(Protocol):
    """What a provider of the workgroups capability implements."""

    async def create_swarm(
        self,
        repo: str,
        task_id: str | None = None,
        coding_agent_session: str = "",
        workers: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]: ...

    async def list_swarms(
        self, repo: str | None = None, status: str | None = None
    ) -> dict[str, Any]: ...

    async def get_swarm_detail(self, swarm_id: str) -> dict[str, Any]: ...

    async def update_worker_status(
        self,
        swarm_id: str,
        worker_name: str,
        status: str,
        pr_number: int | None = None,
    ) -> dict[str, Any]: ...

    async def broadcast_to_swarm(
        self, swarm_id: str, from_entity: str, message: str
    ) -> dict[str, Any]: ...

    async def complete_swarm(self, swarm_id: str) -> dict[str, Any]: ...


def register_workgroups_tools(registry: ToolRegistry, provider: WorkgroupsProvider) -> None:
    """Register backbone swarm management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="create_swarm",
            description=(
                "Create a new coding swarm in the backbone with its lead session and optional "
                "worker registrations."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Repository the swarm is working in",
                    },
                    "task_id": {
                        "type": "string",
                        "description": "Optional task or issue identifier associated with the swarm",
                    },
                    "coding_agent_session": {
                        "type": "string",
                        "description": "Lead coding-agent session for the swarm",
                    },
                    "workers": {
                        "type": "array",
                        "description": "Optional worker registrations to attach during swarm creation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "role": {
                                    "type": "string",
                                    "enum": sorted(_SWARM_WORKER_ROLES),
                                },
                                "branch": {"type": "string"},
                                "worktree_path": {"type": "string"},
                                "session": {"type": "string"},
                            },
                            "required": list(_WORKER_REQUIRED_FIELDS),
                        },
                    },
                },
                "required": ["repo", "coding_agent_session"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.create_swarm,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="list_swarms",
            description=(
                "List swarms known to the backbone, optionally filtered by repository or "
                "swarm status/phase."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Optional repository filter",
                    },
                    "status": {
                        "type": "string",
                        "description": "Optional swarm phase/status filter",
                        "enum": sorted(_SWARM_PHASES),
                    },
                },
            },
            category=ToolCategory.BACKEND,
        ),
        provider.list_swarms,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_swarm_detail",
            description="Get full detail for a swarm, including worker status and progress.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "swarm_id": {
                        "type": "string",
                        "description": "Swarm identifier to inspect",
                    }
                },
                "required": ["swarm_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_swarm_detail,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="update_worker_status",
            description=("Update a swarm worker's status and return the refreshed swarm detail."),
            parameters_schema={
                "type": "object",
                "properties": {
                    "swarm_id": {
                        "type": "string",
                        "description": "Swarm identifier",
                    },
                    "worker_name": {
                        "type": "string",
                        "description": "Worker name within the swarm",
                    },
                    "status": {
                        "type": "string",
                        "description": "New worker status",
                        "enum": sorted(_SWARM_WORKER_STATUSES),
                    },
                    "pr_number": {
                        "type": "integer",
                        "description": "Optional pull request number associated with the worker",
                    },
                },
                "required": ["swarm_id", "worker_name", "status"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.update_worker_status,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="broadcast_to_swarm",
            description="Broadcast a lead message to every worker session in a swarm.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "swarm_id": {
                        "type": "string",
                        "description": "Swarm identifier",
                    },
                    "from_entity": {
                        "type": "string",
                        "description": "Entity sending the broadcast",
                    },
                    "message": {
                        "type": "string",
                        "description": "Broadcast message content",
                    },
                },
                "required": ["swarm_id", "from_entity", "message"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.broadcast_to_swarm,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="complete_swarm",
            description="Mark a swarm as completed and return the final swarm state.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "swarm_id": {
                        "type": "string",
                        "description": "Swarm identifier to complete",
                    }
                },
                "required": ["swarm_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.complete_swarm,
    )
