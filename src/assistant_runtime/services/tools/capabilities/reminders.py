"""Reminders: timed items on the user's schedule.

Add, remove or toggle an item.

A provider implements ``RemindersProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class RemindersProvider(Protocol):
    """What a provider of the reminders capability implements."""

    async def add_schedule_item(self, time: str, title: str) -> dict[str, Any]: ...

    async def remove_schedule_item(self, item_id: str) -> dict[str, Any]: ...

    async def toggle_schedule_item(self, item_id: str) -> dict[str, Any]: ...


def register_reminders_tools(registry: ToolRegistry, provider: RemindersProvider) -> None:
    """Register all schedule management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="add_schedule_item",
            description=(
                "Add a personal schedule item for today. "
                "Specify the time (HH:MM format) and a title/description."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Scheduled time in HH:MM format (e.g. '14:30')",
                    },
                    "title": {
                        "type": "string",
                        "description": "Description of the schedule item",
                    },
                },
                "required": ["time", "title"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.add_schedule_item,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="remove_schedule_item",
            description=(
                "Remove a personal schedule item by its ID. "
                "Only personal items can be removed (not heartbeat or cron items)."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The schedule item ID to remove",
                    },
                },
                "required": ["item_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.remove_schedule_item,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="toggle_schedule_item",
            description=(
                "Toggle the done/not-done state of a schedule item. "
                "Reads the current state and sends the opposite value."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The schedule item ID to toggle",
                    },
                },
                "required": ["item_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.toggle_schedule_item,
    )

    logger.info("Registered schedule management tools", count=3)
