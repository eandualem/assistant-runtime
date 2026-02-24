"""Tests for meeting room management tools."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lovely_assistant.services.tools._meeting_tools import (
    create_meeting_room,
    list_meeting_rooms,
    register_meeting_tools,
    send_meeting_message,
    update_meeting_state,
)
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.config import ToolConfig

MODULE = "lovely_assistant.services.tools._meeting_tools"


# ---------------------------------------------------------------------------
# TestCreateMeetingRoom
# ---------------------------------------------------------------------------


class TestCreateMeetingRoom:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            201,
            {
                "id": "room-1",
                "topic": "Sprint planning",
                "state": "active",
                "participants": ["leo", "ike"],
            },
        )

        result = await create_meeting_room(
            title="Sprint planning",
            participants=["leo", "ike"],
        )
        assert result["success"] is True
        assert result["id"] == "room-1"
        assert result["title"] == "Sprint planning"
        assert result["state"] == "active"

    async def test_missing_title(self):
        result = await create_meeting_room(title="", participants=["leo"])
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_empty_participants(self):
        result = await create_meeting_room(title="Test", participants=[])
        assert result["success"] is False
        assert "participant" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal Server Error"})

        result = await create_meeting_room(title="Test", participants=["leo"])
        assert result["success"] is False
        assert "500" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_network_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await create_meeting_room(title="Test", participants=["leo"])
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_with_description(self, mock_req):
        mock_req.return_value = (
            201,
            {"id": "room-2", "topic": "Review", "state": "active", "participants": ["leo"]},
        )

        result = await create_meeting_room(
            title="Review",
            participants=["leo"],
            description="Architecture review",
        )
        assert result["success"] is True

        call_kwargs = mock_req.call_args
        json_body = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert json_body["context"] == "Architecture review"


# ---------------------------------------------------------------------------
# TestListMeetingRooms
# ---------------------------------------------------------------------------


class TestListMeetingRooms:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "id": "room-1",
                        "topic": "Sprint",
                        "state": "active",
                        "participants": ["leo", "ike"],
                        "created_at": "2026-02-19T10:00:00Z",
                    },
                ],
                "total": 1,
            },
        )

        result = await list_meeting_rooms()
        assert result["success"] is True
        assert result["count"] == 1
        assert result["rooms"][0]["id"] == "room-1"
        assert result["rooms"][0]["title"] == "Sprint"
        assert result["rooms"][0]["participant_count"] == 2

    @patch(f"{MODULE}.backbone_request")
    async def test_with_state_filter(self, mock_req):
        mock_req.return_value = (200, {"items": [], "total": 0})

        await list_meeting_rooms(state="closed")

        call_kwargs = mock_req.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params == {"state": "closed"}

    @patch(f"{MODULE}.backbone_request")
    async def test_empty_list(self, mock_req):
        mock_req.return_value = (200, {"items": [], "total": 0})

        result = await list_meeting_rooms()
        assert result["success"] is True
        assert result["count"] == 0
        assert result["rooms"] == []


# ---------------------------------------------------------------------------
# TestSendMeetingMessage
# ---------------------------------------------------------------------------


class TestSendMeetingMessage:
    @patch(f"{MODULE}.backbone_request")
    async def test_broadcast(self, mock_req):
        mock_req.return_value = (200, {"ok": True})

        result = await send_meeting_message("room-1", "Hello everyone")
        assert result["success"] is True
        assert result["directed"] is False

        call_args = mock_req.call_args
        assert "/broadcast" in call_args[0][1]

    @patch(f"{MODULE}.backbone_request")
    async def test_directed(self, mock_req):
        mock_req.return_value = (200, {"ok": True})

        result = await send_meeting_message("room-1", "Hey Leo", target="leo")
        assert result["success"] is True
        assert result["directed"] is True

        call_args = mock_req.call_args
        assert "/directed" in call_args[0][1]
        json_body = call_args.kwargs.get("json_body") or call_args[1].get("json_body")
        assert json_body["target"] == "leo"

    async def test_empty_message(self):
        result = await send_meeting_message("room-1", "")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (404, {"detail": "Room not found"})

        result = await send_meeting_message("room-999", "hello")
        assert result["success"] is False
        assert "404" in result["error"]


# ---------------------------------------------------------------------------
# TestUpdateMeetingState
# ---------------------------------------------------------------------------


class TestUpdateMeetingState:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (200, {"id": "room-1", "state": "closed"})

        result = await update_meeting_state("room-1", "closed")
        assert result["success"] is True
        assert result["state"] == "closed"
        assert result["room_id"] == "room-1"

    async def test_invalid_state(self):
        result = await update_meeting_state("room-1", "destroyed")
        assert result["success"] is False
        assert "Invalid state" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal error"})

        result = await update_meeting_state("room-1", "closed")
        assert result["success"] is False
        assert "500" in result["error"]

    @pytest.mark.parametrize("state", ["active", "paused", "closed"])
    @patch(f"{MODULE}.backbone_request")
    async def test_all_valid_states(self, mock_req, state):
        mock_req.return_value = (200, {"id": "room-1", "state": state})

        result = await update_meeting_state("room-1", state)
        assert result["success"] is True
        assert result["state"] == state


# ---------------------------------------------------------------------------
# TestRegisterMeetingTools
# ---------------------------------------------------------------------------


class TestRegisterMeetingTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_meeting_tools(registry)

        names = registry.get_tool_names()
        assert "create_meeting_room" in names
        assert "list_meeting_rooms" in names
        assert "send_meeting_message" in names
        assert "update_meeting_state" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_meeting_tools(registry)
        assert len(registry._backend_definitions) == 4

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_meeting_tools(registry)
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_meeting_tools(registry)
        for name in [
            "create_meeting_room",
            "list_meeting_rooms",
            "send_meeting_message",
            "update_meeting_state",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
