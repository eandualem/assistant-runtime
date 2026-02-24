"""Meeting room management tools -- create, list, message, and update meeting rooms."""

from __future__ import annotations

from typing import Any

from loguru import logger

from lovely_assistant.services.tools._backbone_client import backbone_error, backbone_request
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

VALID_ROOM_STATES = {"active", "paused", "closed"}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def create_meeting_room(
    title: str,
    participants: list[str],
    description: str = "",
    moderator: str = "elias",
) -> dict[str, Any]:
    """Create a new meeting room for structured multi-agent discussions."""
    if not title or not title.strip():
        return {"error": "Title cannot be empty", "success": False}

    if not participants:
        return {"error": "At least one participant is required", "success": False}

    status, data = await backbone_request(
        "POST",
        "/api/rooms",
        json_body={
            "topic": title.strip(),
            "participants": participants,
            "context": description,
        },
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
        "title": data.get("topic", title.strip()),
        "state": data.get("state", "active"),
        "participants": data.get("participants", participants),
        "success": True,
    }


async def list_meeting_rooms(state: str = "") -> dict[str, Any]:
    """List meeting rooms, optionally filtered by state."""
    params: dict[str, str] = {}
    if state:
        params["state"] = state

    status, data = await backbone_request("GET", "/api/rooms", params=params or None)

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status != 200:
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    items = data.get("items", [])
    rooms = [
        {
            "id": room.get("id"),
            "title": room.get("topic", ""),
            "state": room.get("state", "unknown"),
            "participant_count": len(room.get("participants", [])),
            "created_at": room.get("created_at"),
        }
        for room in items
    ]

    return {"rooms": rooms, "count": len(rooms), "success": True}


async def send_meeting_message(
    room_id: str,
    message: str,
    target: str = "",
) -> dict[str, Any]:
    """Send a message in a meeting room (broadcast or directed to a participant)."""
    if not message or not message.strip():
        return {"error": "Message cannot be empty", "success": False}

    if target:
        status, data = await backbone_request(
            "POST",
            f"/api/rooms/{room_id}/directed",
            json_body={"target": target, "content": message.strip()},
        )
    else:
        status, data = await backbone_request(
            "POST",
            f"/api/rooms/{room_id}/broadcast",
            json_body={"content": message.strip()},
        )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status not in (200, 201):
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {
        "room_id": room_id,
        "directed": bool(target),
        "success": True,
    }


async def update_meeting_state(room_id: str, state: str) -> dict[str, Any]:
    """Update the state of a meeting room (active, paused, closed)."""
    if state not in VALID_ROOM_STATES:
        return {
            "error": f"Invalid state '{state}'. Must be one of: {', '.join(sorted(VALID_ROOM_STATES))}",
            "success": False,
        }

    status, data = await backbone_request(
        "PATCH",
        f"/api/rooms/{room_id}",
        json_body={"state": state},
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status != 200:
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {
        "room_id": room_id,
        "state": state,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_meeting_tools(registry: ToolRegistry) -> None:
    """Register all meeting room management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="create_meeting_room",
            description=(
                "Create a new meeting room for structured multi-agent discussions. "
                "Specify a title, list of participant session names, optional description, "
                "and moderator (defaults to 'elias')."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Meeting room title/topic",
                    },
                    "participants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of participant session names (e.g. ['leo', 'ike'])",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional meeting description or context",
                        "default": "",
                    },
                    "moderator": {
                        "type": "string",
                        "description": "Moderator name (defaults to 'elias')",
                        "default": "elias",
                    },
                },
                "required": ["title", "participants"],
            },
            category=ToolCategory.BACKEND,
        ),
        create_meeting_room,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="list_meeting_rooms",
            description=(
                "List meeting rooms, optionally filtered by state "
                "(active, paused, closed). Returns room ID, title, state, "
                "participant count, and creation time."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["active", "paused", "closed", ""],
                        "description": "Optional state filter",
                        "default": "",
                    },
                },
            },
            category=ToolCategory.BACKEND,
        ),
        list_meeting_rooms,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="send_meeting_message",
            description=(
                "Send a message in a meeting room. If 'target' is specified, sends a "
                "directed message to that participant. Otherwise broadcasts to all participants."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The meeting room ID",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message text to send",
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target participant for directed message",
                        "default": "",
                    },
                },
                "required": ["room_id", "message"],
            },
            category=ToolCategory.BACKEND,
        ),
        send_meeting_message,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="update_meeting_state",
            description=(
                "Update the state of a meeting room. Valid states are "
                "'active', 'paused', and 'closed'."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "room_id": {
                        "type": "string",
                        "description": "The meeting room ID",
                    },
                    "state": {
                        "type": "string",
                        "enum": ["active", "paused", "closed"],
                        "description": "New room state",
                    },
                },
                "required": ["room_id", "state"],
            },
            category=ToolCategory.BACKEND,
        ),
        update_meeting_state,
    )

    logger.info("Registered meeting room management tools", count=4)
