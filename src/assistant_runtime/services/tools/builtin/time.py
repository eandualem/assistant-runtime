"""``get_time`` — the current UTC time."""

from __future__ import annotations

from datetime import UTC, datetime

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


async def get_time() -> str:
    """Get the current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def register_time_tools(registry: ToolRegistry) -> None:
    registry.register_backend_tool(
        ToolDefinition(
            name="get_time",
            description="Get the current UTC time in ISO format.",
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        get_time,
    )
