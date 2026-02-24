"""Schedule management tools -- add, remove, and toggle personal schedule items."""

from __future__ import annotations

from typing import Any

from loguru import logger

from lovely_assistant.services.tools._backbone_client import backbone_error, backbone_request
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def add_schedule_item(time: str, title: str) -> dict[str, Any]:
    """Add a personal schedule item for today."""
    if not time or not time.strip():
        return {"error": "Time cannot be empty", "success": False}

    if not title or not title.strip():
        return {"error": "Title cannot be empty", "success": False}

    status, data = await backbone_request(
        "POST",
        "/api/schedule/personal",
        json_body={"time": time.strip(), "description": title.strip()},
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status not in (200, 201):
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {
        "id": data.get("id"),
        "time": data.get("time", time.strip()),
        "title": data.get("description", title.strip()),
        "success": True,
    }


async def remove_schedule_item(item_id: str) -> dict[str, Any]:
    """Remove a personal schedule item."""
    status, data = await backbone_request(
        "DELETE",
        f"/api/schedule/personal/{item_id}",
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status == 404:
        return {"error": f"Schedule item '{item_id}' not found", "success": False}

    if status not in (200, 204):
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {"item_id": item_id, "removed": True, "success": True}


async def toggle_schedule_item(item_id: str) -> dict[str, Any]:
    """Toggle the done state of a schedule item."""
    # First, get today's schedule to find the item's current done state
    get_status, get_data = await backbone_request("GET", "/api/schedule/today")

    if get_status == -1:
        return {"error": backbone_error(get_data), "success": False}

    if get_status != 200:
        return {
            "error": f"Backbone API error ({get_status}): {get_data.get('detail', 'Unknown error')}",
            "success": False,
        }

    # Find the item in today's schedule
    items = get_data.get("items", [])
    current_item = None
    for item in items:
        if item.get("id") == item_id:
            current_item = item
            break

    if current_item is None:
        return {"error": f"Schedule item '{item_id}' not found", "success": False}

    new_done = not current_item.get("done", False)

    status, data = await backbone_request(
        "PATCH",
        f"/api/schedule/today/{item_id}",
        json_body={"done": new_done},
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status == 404:
        return {"error": f"Schedule item '{item_id}' not found", "success": False}

    if status != 200:
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {
        "item_id": item_id,
        "done": data.get("done", new_done),
        "success": True,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_schedule_tools(registry: ToolRegistry) -> None:
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
        add_schedule_item,
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
        remove_schedule_item,
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
        toggle_schedule_item,
    )

    logger.info("Registered schedule management tools", count=3)
