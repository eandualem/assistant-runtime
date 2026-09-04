"""Tests for backbone telemetry and delivery status tools."""

from __future__ import annotations

from unittest.mock import patch

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.activity import register_activity_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.backbone.activity import (
    BackboneActivity,
    get_activity_timeline,
    get_agent_activity,
    get_delivery_status,
    get_failed_deliveries,
    get_recent_deliveries,
)

MODULE = "assistant_runtime.services.tools.providers.backbone.activity"


class TestGetDeliveryStatus:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {"total": 10, "delivered": 7, "failed": 1, "deferred": 1, "offline": 1},
        )

        result = await get_delivery_status()
        assert result["success"] is True
        assert result["total"] == 10
        assert result["delivered"] == 7
        assert result["failed"] == 1
        assert result["deferred"] == 1
        assert result["offline"] == 1

    @patch(f"{MODULE}.backbone_request")
    async def test_network_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await get_delivery_status()
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal error"})

        result = await get_delivery_status()
        assert result["success"] is False
        assert "500" in result["error"]


class TestGetRecentDeliveries:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "id": 1,
                        "issue_number": 100,
                        "target_entity": "coding-agent",
                        "session_name": "assistant-runtime",
                        "outcome": "delivered",
                    }
                ],
                "total": 1,
            },
        )

        result = await get_recent_deliveries(limit=5)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["total"] == 1
        assert result["deliveries"][0]["issue_number"] == 100

        mock_req.assert_awaited_once_with("GET", "/api/deliveries", params={"limit": "5"})

    async def test_invalid_limit(self):
        result = await get_recent_deliveries(limit=0)
        assert result["success"] is False
        assert "limit" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_unexpected_shape(self, mock_req):
        mock_req.return_value = (200, {"items": "not-a-list", "total": 1})

        result = await get_recent_deliveries()
        assert result["success"] is False
        assert result["error"] == "Unexpected backbone response format"


class TestGetFailedDeliveries:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "id": 2,
                        "issue_number": 101,
                        "target_entity": "coding-agent",
                        "session_name": "assistant-runtime",
                        "outcome": "offline",
                    }
                ],
                "total": 1,
            },
        )

        result = await get_failed_deliveries(limit=3)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["deliveries"][0]["outcome"] == "offline"
        mock_req.assert_awaited_once_with("GET", "/api/deliveries/failed", params={"limit": "3"})

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (503, {"detail": "Service unavailable"})

        result = await get_failed_deliveries()
        assert result["success"] is False
        assert "503" in result["error"]


class TestGetAgentActivity:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "id": 10,
                        "session": "agent-backbone",
                        "event": "tool.finished",
                        "ts": 1773407310.573,
                        "received_at": "2026-03-13T13:08:32.344807Z",
                    }
                ],
                "total": 1,
            },
        )

        result = await get_agent_activity("  agent-backbone  ", limit=2, since=123.5)
        assert result["success"] is True
        assert result["session_name"] == "agent-backbone"
        assert result["count"] == 1
        assert result["events"][0]["event"] == "tool.finished"

        mock_req.assert_awaited_once_with(
            "GET",
            "/api/agents/agent-backbone/activity",
            params={"limit": "2", "since": "123.5"},
        )

    async def test_empty_session_name(self):
        result = await get_agent_activity("")
        assert result["success"] is False
        assert "session_name" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_not_found(self, mock_req):
        mock_req.return_value = (404, {"detail": "Unknown session"})

        result = await get_agent_activity("missing-agent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestGetActivityTimeline:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "ts": 1773409178.493482,
                        "type": "delivery",
                        "entity": "coding-agent",
                        "summary": "#727 -> assistant-runtime (delivered)",
                    }
                ],
                "total": 8,
            },
        )

        result = await get_activity_timeline(limit=4, offset=2)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["total"] == 8
        assert result["offset"] == 2
        assert result["events"][0]["type"] == "delivery"

        mock_req.assert_awaited_once_with(
            "GET",
            "/api/activity/timeline",
            params={"limit": "4", "offset": "2"},
        )

    async def test_negative_offset(self):
        result = await get_activity_timeline(offset=-1)
        assert result["success"] is False
        assert "offset" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_network_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await get_activity_timeline()
        assert result["success"] is False
        assert "Connection refused" in result["error"]


class TestRegisterTelemetryTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_activity_tools(registry, BackboneActivity())

        names = registry.get_tool_names()
        assert "get_delivery_status" in names
        assert "get_recent_deliveries" in names
        assert "get_failed_deliveries" in names
        assert "get_agent_activity" in names
        assert "get_activity_timeline" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_activity_tools(registry, BackboneActivity())
        assert len(registry._backend_definitions) == 5

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_activity_tools(registry, BackboneActivity())
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_activity_tools(registry, BackboneActivity())
        for name in [
            "get_delivery_status",
            "get_recent_deliveries",
            "get_failed_deliveries",
            "get_agent_activity",
            "get_activity_timeline",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
