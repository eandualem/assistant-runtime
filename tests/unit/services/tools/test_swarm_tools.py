"""Tests for backbone swarm management tools."""

from __future__ import annotations

from unittest.mock import patch

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._swarm_tools import (
    broadcast_to_swarm,
    complete_swarm,
    create_swarm,
    get_swarm_detail,
    list_swarms,
    register_swarm_tools,
    update_worker_status,
)
from lovely_assistant.services.tools.config import ToolConfig

MODULE = "lovely_assistant.services.tools._swarm_tools"

SAMPLE_WORKER = {
    "name": "coder-1",
    "role": "coder",
    "branch": "feature/swarm",
    "worktree_path": "/tmp/swarm-coder-1",
    "session": "lovely-assistant-coder-1",
}

SAMPLE_SWARM_DETAIL = {
    "swarm_id": "swarm-1",
    "repo": "lovely-assistant",
    "task_id": "742",
    "coding_agent_session": "lovely-assistant",
    "phase": "working",
    "created_at": "2026-03-13T18:00:00Z",
    "completed_at": None,
    "worker_count": 1,
    "progress": {"total": 1, "done": 0, "failed": 0},
    "workers": [
        {
            "worker_id": "worker-1",
            "swarm_id": "swarm-1",
            "name": "coder-1",
            "role": "coder",
            "branch": "feature/swarm",
            "worktree_path": "/tmp/swarm-coder-1",
            "session": "lovely-assistant-coder-1",
            "status": "working",
            "pr_number": None,
            "summary": None,
            "failure_reason": None,
            "completed_at": None,
            "created_at": "2026-03-13T18:00:00Z",
            "updated_at": "2026-03-13T18:01:00Z",
        }
    ],
    "workers_by_role": {"coder": []},
    "phase_history": [],
}


class TestCreateSwarm:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (201, {"swarm_id": "swarm-1"})

        result = await create_swarm(
            " lovely-assistant ",
            task_id="742",
            coding_agent_session=" coding-agent ",
            workers=[SAMPLE_WORKER],
        )

        assert result["success"] is True
        assert result["swarm_id"] == "swarm-1"
        assert result["repo"] == "lovely-assistant"
        assert result["coding_agent_session"] == "coding-agent"
        assert result["worker_count"] == 1
        mock_req.assert_awaited_once_with(
            "POST",
            "/api/swarms",
            json_body={
                "repo": "lovely-assistant",
                "task_id": "742",
                "coding_agent_session": "coding-agent",
                "workers": [SAMPLE_WORKER],
            },
        )

    async def test_invalid_worker_role(self):
        result = await create_swarm(
            "lovely-assistant",
            coding_agent_session="coding-agent",
            workers=[{**SAMPLE_WORKER, "role": "manager"}],
        )

        assert result["success"] is False
        assert "workers[0].role" in result["error"]

    @patch(f"{MODULE}.backbone_request")
    async def test_network_error(self, mock_req):
        mock_req.return_value = (-1, {"message": "Connection refused"})

        result = await create_swarm("lovely-assistant", coding_agent_session="coding-agent")

        assert result["success"] is False
        assert "Connection refused" in result["error"]


class TestListSwarms:
    @patch(f"{MODULE}.backbone_request")
    async def test_success_with_filters(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "swarm_id": "swarm-1",
                        "repo": "lovely-assistant",
                        "coding_agent_session": "coding-agent",
                        "phase": "working",
                        "created_at": "2026-03-13T18:00:00Z",
                    }
                ],
                "total": 1,
            },
        )

        result = await list_swarms(repo=" lovely-assistant ", status="working")

        assert result["success"] is True
        assert result["count"] == 1
        assert result["swarms"][0]["swarm_id"] == "swarm-1"
        mock_req.assert_awaited_once_with(
            "GET",
            "/api/swarms",
            params={"repo": "lovely-assistant", "phase": "working"},
        )

    @patch(f"{MODULE}.backbone_request")
    async def test_success_without_filters(self, mock_req):
        mock_req.return_value = (200, {"items": [], "total": 0})

        result = await list_swarms()

        assert result["success"] is True
        assert result["count"] == 0
        mock_req.assert_awaited_once_with("GET", "/api/swarms", params=None)

    async def test_invalid_status(self):
        result = await list_swarms(status="queued")

        assert result["success"] is False
        assert "status must be one of" in result["error"]


class TestGetSwarmDetail:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (200, SAMPLE_SWARM_DETAIL)

        result = await get_swarm_detail(" swarm-1 ")

        assert result["success"] is True
        assert result["swarm"]["swarm_id"] == "swarm-1"
        mock_req.assert_awaited_once_with("GET", "/api/swarms/swarm-1")

    @patch(f"{MODULE}.backbone_request")
    async def test_not_found(self, mock_req):
        mock_req.return_value = (404, {"detail": "Missing swarm"})

        result = await get_swarm_detail("swarm-404")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestUpdateWorkerStatus:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (200, SAMPLE_SWARM_DETAIL)

        result = await update_worker_status(
            " swarm-1 ",
            " coder-1 ",
            "done",
            pr_number=42,
        )

        assert result["success"] is True
        assert result["worker_name"] == "coder-1"
        assert result["status"] == "done"
        mock_req.assert_awaited_once_with(
            "POST",
            "/api/swarms/swarm-1/workers/coder-1/status",
            json_body={"status": "done", "pr_number": 42},
        )

    async def test_invalid_status(self):
        result = await update_worker_status("swarm-1", "coder-1", "queued")

        assert result["success"] is False
        assert "status must be one of" in result["error"]

    async def test_invalid_pr_number(self):
        result = await update_worker_status("swarm-1", "coder-1", "done", pr_number=0)

        assert result["success"] is False
        assert "pr_number" in result["error"]


class TestBroadcastToSwarm:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {"ok": True, "message_id": 99, "delivered": 2, "failed": 0, "total": 2},
        )

        result = await broadcast_to_swarm(
            " swarm-1 ", " bell-wf ", " Investigate the failing test "
        )

        assert result["success"] is True
        assert result["ok"] is True
        assert result["delivered"] == 2
        mock_req.assert_awaited_once_with(
            "POST",
            "/api/swarms/swarm-1/broadcast",
            json_body={"from_entity": "bell-wf", "message": "Investigate the failing test"},
        )

    async def test_empty_message(self):
        result = await broadcast_to_swarm("swarm-1", "bell-wf", "")

        assert result["success"] is False
        assert "message" in result["error"]


class TestCompleteSwarm:
    @patch(f"{MODULE}.backbone_request")
    async def test_success(self, mock_req):
        completed = {
            **SAMPLE_SWARM_DETAIL,
            "phase": "cleaned_up",
            "completed_at": "2026-03-13T20:00:00Z",
        }
        mock_req.return_value = (200, completed)

        result = await complete_swarm(" swarm-1 ")

        assert result["success"] is True
        assert result["swarm"]["phase"] == "cleaned_up"
        mock_req.assert_awaited_once_with("DELETE", "/api/swarms/swarm-1")

    @patch(f"{MODULE}.backbone_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (409, {"detail": "Swarm is not merged"})

        result = await complete_swarm("swarm-1")

        assert result["success"] is False
        assert "409" in result["error"]


class TestRegisterSwarmTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_swarm_tools(registry)

        names = registry.get_tool_names()
        assert "create_swarm" in names
        assert "list_swarms" in names
        assert "get_swarm_detail" in names
        assert "update_worker_status" in names
        assert "broadcast_to_swarm" in names
        assert "complete_swarm" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_swarm_tools(registry)
        assert len(registry._backend_definitions) == 6

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_swarm_tools(registry)
        for definition in registry._backend_definitions.values():
            assert definition.category == "backend"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_swarm_tools(registry)
        for name in [
            "create_swarm",
            "list_swarms",
            "get_swarm_detail",
            "update_worker_status",
            "broadcast_to_swarm",
            "complete_swarm",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
