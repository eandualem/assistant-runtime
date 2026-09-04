"""Rooms: shared conversations between agents (and people).

Create a room, list rooms, send a message into one, change its state.

A provider implements ``RoomsProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class RoomsProvider(Protocol):
    """What a provider of the rooms capability implements."""

    async def create_meeting_room(
        self,
        title: str,
        participants: list[str],
        description: str = "",
        moderator: str = "operator",
    ) -> dict[str, Any]: ...

    async def list_meeting_rooms(self, state: str = "") -> dict[str, Any]: ...

    async def send_meeting_message(
        self,
        room_id: str,
        message: str,
        target: str = "",
    ) -> dict[str, Any]: ...

    async def update_meeting_state(self, room_id: str, state: str) -> dict[str, Any]: ...


def register_rooms_tools(registry: ToolRegistry, provider: RoomsProvider) -> None:
    """Register all meeting room management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="create_meeting_room",
            description=(
                "Create a new meeting room for structured multi-agent discussions. "
                "Specify a title, list of participant session names, optional description, "
                "and moderator (defaults to 'operator')."
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
                        "description": "Moderator name (defaults to 'operator')",
                        "default": "operator",
                    },
                },
                "required": ["title", "participants"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.create_meeting_room,
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
        provider.list_meeting_rooms,
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
        provider.send_meeting_message,
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
        provider.update_meeting_state,
    )

    logger.info("Registered meeting room management tools", count=4)
