"""Activity: what the agents have been doing and whether messages reached them.

A provider implements ``ActivityProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class ActivityProvider(Protocol):
    """What a provider of the activity capability implements."""

    async def get_delivery_status(self) -> dict[str, Any]: ...

    async def get_recent_deliveries(self, limit: int = 20) -> dict[str, Any]: ...

    async def get_failed_deliveries(self, limit: int = 20) -> dict[str, Any]: ...

    async def get_agent_activity(
        self,
        session_name: str,
        limit: int = 20,
        since: float | None = None,
    ) -> dict[str, Any]: ...

    async def get_activity_timeline(self, limit: int = 20, offset: int = 0) -> dict[str, Any]: ...


def register_activity_tools(registry: ToolRegistry, provider: ActivityProvider) -> None:
    """Register backbone telemetry and delivery status tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="get_delivery_status",
            description=(
                "Get backbone delivery health summary: total attempts and counts for "
                "delivered, failed, deferred, and offline outcomes."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        provider.get_delivery_status,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_recent_deliveries",
            description=(
                "List recent backbone delivery attempts with issue number, target entity, "
                "session, outcome, and timestamp."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of delivery records to return (1-500)",
                        "default": 20,
                    }
                },
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_recent_deliveries,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_failed_deliveries",
            description=("List failed, deferred, or offline delivery attempts from the backbone."),
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of failed delivery records to return (1-500)",
                        "default": 20,
                    }
                },
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_failed_deliveries,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_agent_activity",
            description=(
                "Get recent recorded activity for a specific agent session from the backbone."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Agent session name to inspect",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of activity events to return (1-500)",
                        "default": 20,
                    },
                    "since": {
                        "type": "number",
                        "description": "Optional UNIX timestamp lower bound for activity events",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_agent_activity,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_activity_timeline",
            description=(
                "Get the system-wide backbone activity timeline across deliveries, telemetry, "
                "actions, and heartbeats."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of timeline events to return (1-500)",
                        "default": 20,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset into the activity timeline",
                        "default": 0,
                    },
                },
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_activity_timeline,
    )
