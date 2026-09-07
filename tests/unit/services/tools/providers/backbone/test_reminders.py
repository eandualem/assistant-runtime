"""Tests for schedule management tools."""

from __future__ import annotations

from unittest.mock import patch

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.reminders import register_reminders_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.backbone.reminders import (
    BackboneReminders,
    add_schedule_item,
    remove_schedule_item,
    toggle_schedule_item,
)

MODULE = "assistant_runtime.services.tools.providers.backbone.reminders"


# ---------------------------------------------------------------------------
# TestAddScheduleItem
# ---------------------------------------------------------------------------


class TestAddScheduleItem:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            201,
            {"id": "item-1", "time": "14:30", "description": "Team sync"},
        )

        result = await add_schedule_item("14:30", "Team sync")
        assert result["success"] is True
        assert result["id"] == "item-1"
        assert result["time"] == "14:30"
        assert result["title"] == "Team sync"

    async def test_missing_time(self):
        result = await add_schedule_item("", "Team sync")
        assert result["success"] is False
        assert "Time" in result["error"]

    async def test_missing_title(self):
        result = await add_schedule_item("14:30", "")
        assert result["success"] is False
        assert "Title" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal error"})

        result = await add_schedule_item("14:30", "Test")
        assert result["success"] is False
        assert "500" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_network_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await add_schedule_item("14:30", "Test")
        assert result["success"] is False
        assert "Connection refused" in result["error"]


# ---------------------------------------------------------------------------
# TestRemoveScheduleItem
# ---------------------------------------------------------------------------


class TestRemoveScheduleItem:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (200, {"ok": True})

        result = await remove_schedule_item("item-1")
        assert result["success"] is True
        assert result["removed"] is True
        assert result["item_id"] == "item-1"

    @patch(f"{MODULE}.backbone_request")
    async def test_not_found(self, mock_req):
        mock_req.return_value = (404, {"detail": "Not found"})

        result = await remove_schedule_item("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal error"})

        result = await remove_schedule_item("item-1")
        assert result["success"] is False
        assert "500" in result["error"]


# ---------------------------------------------------------------------------
# TestToggleScheduleItem
# ---------------------------------------------------------------------------


class TestToggleScheduleItem:
    @patch(f"{MODULE}.backbone_request")
    async def test_success_toggle_to_done(self, mock_req):
        # First call: GET schedule (item is not done)
        # Second call: PATCH toggle
        mock_req.side_effect = [
            (200, {"items": [{"id": "item-1", "done": False}]}),
            (200, {"id": "item-1", "done": True}),
        ]

        result = await toggle_schedule_item("item-1")
        assert result["success"] is True
        assert result["done"] is True

        # Verify PATCH was called with done=True (opposite of current)
        patch_call = mock_req.call_args_list[1]
        json_body = patch_call.kwargs.get("json_body") or patch_call[1].get("json_body")
        assert json_body["done"] is True

    @patch(f"{MODULE}.backbone_request")
    async def test_success_toggle_to_undone(self, mock_req):
        mock_req.side_effect = [
            (200, {"items": [{"id": "item-1", "done": True}]}),
            (200, {"id": "item-1", "done": False}),
        ]

        result = await toggle_schedule_item("item-1")
        assert result["success"] is True
        assert result["done"] is False

    @patch(f"{MODULE}.backbone_request")
    async def test_not_found_in_schedule(self, mock_req):
        mock_req.return_value = (200, {"items": [{"id": "other-item", "done": False}]})

        result = await toggle_schedule_item("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_get_schedule_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await toggle_schedule_item("item-1")
        assert result["success"] is False

    @patch(f"{MODULE}.backbone_request")
    async def test_patch_api_error(self, mock_req):
        mock_req.side_effect = [
            (200, {"items": [{"id": "item-1", "done": False}]}),
            (500, {"detail": "Internal error"}),
        ]

        result = await toggle_schedule_item("item-1")
        assert result["success"] is False
        assert "500" in result["error"]


# ---------------------------------------------------------------------------
# TestRegisterScheduleTools
# ---------------------------------------------------------------------------


class TestRegisterScheduleTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_reminders_tools(registry, BackboneReminders())

        names = registry.get_tool_names()
        assert "add_schedule_item" in names
        assert "remove_schedule_item" in names
        assert "toggle_schedule_item" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_reminders_tools(registry, BackboneReminders())
        assert len(registry._backend_definitions) == 3

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_reminders_tools(registry, BackboneReminders())
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_reminders_tools(registry, BackboneReminders())
        for name in ["add_schedule_item", "remove_schedule_item", "toggle_schedule_item"]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
