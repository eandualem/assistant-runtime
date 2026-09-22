"""Meeting room management tools -- create, list, message, and update meeting rooms."""

from __future__ import annotations

from typing import Any

from assistant_runtime.services.tools.providers.backbone._client import (
    BackboneRequest,
    backbone_detail,
    backbone_error,
    backbone_request,
)

VALID_ROOM_STATES = {"active", "paused", "closed"}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def create_meeting_room(
    title: str,
    participants: list[str],
    description: str = "",
    moderator: str = "operator",
) -> dict[str, Any]:
    """Create a new meeting room for structured multi-agent discussions."""
    return await BackboneRooms().create_meeting_room(
        title=title, participants=participants, description=description, moderator=moderator
    )


async def list_meeting_rooms(state: str = "") -> dict[str, Any]:
    """List meeting rooms, optionally filtered by state."""
    return await BackboneRooms().list_meeting_rooms(state=state)


async def send_meeting_message(
    room_id: str,
    message: str,
    target: str = "",
) -> dict[str, Any]:
    """Send a message in a meeting room (broadcast or directed to a participant)."""
    return await BackboneRooms().send_meeting_message(
        room_id=room_id, message=message, target=target
    )


async def update_meeting_state(room_id: str, state: str) -> dict[str, Any]:
    """Update the state of a meeting room (active, paused, closed)."""
    return await BackboneRooms().update_meeting_state(room_id=room_id, state=state)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class BackboneRooms:
    """The rooms capability served by this provider (see ``capabilities.rooms``)."""

    def __init__(self, *, request: BackboneRequest | None = None) -> None:
        self._request = request if request is not None else backbone_request

    async def create_meeting_room(
        self,
        title: str,
        participants: list[str],
        description: str = "",
        moderator: str = "operator",
    ) -> dict[str, Any]:
        """Create a new meeting room for structured multi-agent discussions."""
        if not title or not title.strip():
            return {"error": "Title cannot be empty", "success": False}

        if not participants:
            return {"error": "At least one participant is required", "success": False}

        status, data = await self._request(
            "POST",
            "/api/rooms",
            json_body={
                "title": title.strip(),
                "description": description,
                "moderator": moderator,
                "participants": participants,
            },
        )

        if status == -1:
            return {"error": backbone_error(data), "success": False}

        if status not in (200, 201):
            return {
                "error": f"Backbone API error ({status}): {backbone_detail(data)}",
                "success": False,
            }

        return {
            "id": data.get("id"),
            "title": data.get("title", title.strip()),
            "state": data.get("state", "active"),
            "participants": data.get("participants", participants),
            "success": True,
        }

    async def list_meeting_rooms(self, state: str = "") -> dict[str, Any]:
        """List meeting rooms, optionally filtered by state."""
        params: dict[str, str] = {}
        if state:
            params["state"] = state

        status, data = await self._request("GET", "/api/rooms", params=params or None)

        if status == -1:
            return {"error": backbone_error(data), "success": False}

        if status != 200:
            return {
                "error": f"Backbone API error ({status}): {backbone_detail(data)}",
                "success": False,
            }

        items = data.get("items", [])
        rooms = [
            {
                "id": room.get("id"),
                "title": room.get("title", ""),
                "state": room.get("state", "unknown"),
                "participant_count": len(room.get("participants", [])),
                "created_at": room.get("created_at"),
            }
            for room in items
        ]

        return {"rooms": rooms, "count": len(rooms), "success": True}

    async def send_meeting_message(
        self,
        room_id: str,
        message: str,
        target: str = "",
    ) -> dict[str, Any]:
        """Send a message in a meeting room (broadcast or directed to a participant)."""
        if not message or not message.strip():
            return {"error": "Message cannot be empty", "success": False}

        if target:
            status, data = await self._request(
                "POST",
                f"/api/rooms/{room_id}/directed",
                json_body={"target": target, "content": message.strip()},
            )
        else:
            status, data = await self._request(
                "POST",
                f"/api/rooms/{room_id}/broadcast",
                json_body={"content": message.strip()},
            )

        if status == -1:
            return {"error": backbone_error(data), "success": False}

        if status not in (200, 201):
            return {
                "error": f"Backbone API error ({status}): {backbone_detail(data)}",
                "success": False,
            }

        return {
            "room_id": room_id,
            "directed": bool(target),
            "success": True,
        }

    async def update_meeting_state(self, room_id: str, state: str) -> dict[str, Any]:
        """Update the state of a meeting room (active, paused, closed)."""
        if state not in VALID_ROOM_STATES:
            return {
                "error": f"Invalid state '{state}'. Must be one of: {', '.join(sorted(VALID_ROOM_STATES))}",
                "success": False,
            }

        status, data = await self._request(
            "PATCH",
            f"/api/rooms/{room_id}",
            json_body={"state": state},
        )

        if status == -1:
            return {"error": backbone_error(data), "success": False}

        if status != 200:
            return {
                "error": f"Backbone API error ({status}): {backbone_detail(data)}",
                "success": False,
            }

        return {
            "room_id": room_id,
            "state": state,
            "success": True,
        }
