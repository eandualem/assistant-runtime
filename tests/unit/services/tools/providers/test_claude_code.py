"""Tests for plan management tools."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.approvals import register_approvals_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.claude_code import (
    StateFileApprovals,
    _get_agent_state,
    _read_all_state_files,
    approve_plan,
    list_agent_plans,
    reject_plan,
)

MODULE = "assistant_runtime.services.tools.providers.claude_code"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR to a temp directory."""
    monkeypatch.setattr(f"{MODULE}.STATE_DIR", tmp_path)
    return tmp_path


def _write_state(state_dir, session_name, state_data):
    """Helper to write a state file."""
    path = state_dir / f"{session_name}.json"
    path.write_text(json.dumps(state_data))


# ---------------------------------------------------------------------------
# TestReadAllStateFiles
# ---------------------------------------------------------------------------


class TestReadAllStateFiles:
    def test_reads_valid_files(self, state_dir):
        _write_state(state_dir, "leo", {"state": "idle"})
        _write_state(state_dir, "ike", {"state": "processing"})

        results = _read_all_state_files()
        assert len(results) == 2
        names = {name for name, _ in results}
        assert "leo" in names
        assert "ike" in names

    def test_skips_invalid_json(self, state_dir):
        _write_state(state_dir, "leo", {"state": "idle"})
        (state_dir / "broken.json").write_text("not json {{{")

        results = _read_all_state_files()
        assert len(results) == 1

    def test_empty_directory(self, state_dir):
        results = _read_all_state_files()
        assert results == []

    def test_nonexistent_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(f"{MODULE}.STATE_DIR", tmp_path / "nonexistent")
        results = _read_all_state_files()
        assert results == []


# ---------------------------------------------------------------------------
# TestGetAgentState
# ---------------------------------------------------------------------------


class TestGetAgentState:
    def test_valid_state(self, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting", "plan_title": "Test plan"})

        result = _get_agent_state("leo")
        assert result is not None
        assert result["state"] == "plan_waiting"

    def test_missing_file(self, state_dir):
        result = _get_agent_state("nonexistent")
        assert result is None

    def test_invalid_json(self, state_dir):
        (state_dir / "broken.json").write_text("not json")
        result = _get_agent_state("broken")
        assert result is None


# ---------------------------------------------------------------------------
# TestListAgentPlans
# ---------------------------------------------------------------------------


class TestListAgentPlans:
    async def test_has_pending_plans(self, state_dir):
        _write_state(
            state_dir,
            "leo",
            {
                "state": "plan_waiting",
                "plan_title": "Implement feature X",
                "plan_file": "/path/to/plan.md",
                "ts": int(time.time()) - 300,  # 5 minutes ago
            },
        )
        _write_state(state_dir, "ike", {"state": "idle"})

        result = await list_agent_plans()
        assert result["success"] is True
        assert result["count"] == 1
        assert result["plans"][0]["session_name"] == "leo"
        assert result["plans"][0]["plan_title"] == "Implement feature X"
        assert result["plans"][0]["plan_file"] == "/path/to/plan.md"
        assert result["plans"][0]["waiting_since"] is not None

    async def test_no_pending_plans(self, state_dir):
        _write_state(state_dir, "leo", {"state": "idle"})
        _write_state(state_dir, "ike", {"state": "processing"})

        result = await list_agent_plans()
        assert result["success"] is True
        assert result["count"] == 0
        assert result["plans"] == []

    async def test_malformed_state_files(self, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting", "plan_title": "Test"})
        (state_dir / "broken.json").write_text("not json")

        result = await list_agent_plans()
        assert result["success"] is True
        assert result["count"] == 1

    async def test_mixed_states(self, state_dir):
        _write_state(
            state_dir,
            "leo",
            {"state": "plan_waiting", "plan_title": "Plan A", "ts": int(time.time())},
        )
        _write_state(state_dir, "ike", {"state": "idle"})
        _write_state(
            state_dir,
            "ada",
            {"state": "plan_waiting", "plan_title": "Plan B", "ts": int(time.time())},
        )
        _write_state(state_dir, "feynman", {"state": "processing"})

        result = await list_agent_plans()
        assert result["count"] == 2
        names = {p["session_name"] for p in result["plans"]}
        assert names == {"leo", "ada"}

    async def test_waiting_since_hours(self, state_dir):
        _write_state(
            state_dir,
            "leo",
            {
                "state": "plan_waiting",
                "plan_title": "Plan",
                "ts": int(time.time()) - 7200,  # 2 hours ago
            },
        )

        result = await list_agent_plans()
        assert "h" in result["plans"][0]["waiting_since"]


# ---------------------------------------------------------------------------
# TestApprovePlan
# ---------------------------------------------------------------------------


class TestApprovePlan:
    @patch(f"{MODULE}._run_command")
    async def test_success(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting", "plan_title": "Test"})
        # has-session, send-keys
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]

        result = await approve_plan("leo")
        assert result["success"] is True
        assert result["approved"] is True
        assert result["session_name"] == "leo"

        # Verify the Shift+Tab escape sequence was sent
        send_call = mock_run.call_args_list[1]
        args = send_call[0][0]
        assert "\\e[Z" in args

    async def test_not_in_plan_waiting(self, state_dir):
        _write_state(state_dir, "leo", {"state": "idle"})

        result = await approve_plan("leo")
        assert result["success"] is False
        assert "not in plan_waiting" in result["error"]

    async def test_invalid_session_name(self, state_dir):
        result = await approve_plan("-invalid")
        assert result["success"] is False

    @patch(f"{MODULE}._run_command")
    async def test_session_does_not_exist(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        mock_run.return_value = (1, "", "session not found")

        result = await approve_plan("leo")
        assert result["success"] is False
        assert "does not exist" in result["error"]

    async def test_no_state_file(self, state_dir):
        result = await approve_plan("ghost")
        assert result["success"] is False
        assert "not in plan_waiting" in result["error"]

    @patch(f"{MODULE}._run_command")
    async def test_send_keys_failure(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        mock_run.side_effect = [
            (0, "", ""),  # has-session
            (1, "", "send-keys failed"),  # send-keys
        ]

        result = await approve_plan("leo")
        assert result["success"] is False
        assert "Failed to send" in result["error"]


# ---------------------------------------------------------------------------
# TestRejectPlan
# ---------------------------------------------------------------------------


class TestRejectPlan:
    @patch(f"{MODULE}._run_command")
    async def test_success(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        # has-session, send-keys -l, send-keys Enter
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await reject_plan("leo", "Needs more detail on testing approach")
        assert result["success"] is True
        assert result["rejected"] is True
        assert result["reason"] == "Needs more detail on testing approach"

        # Verify the reason was sent as text
        text_call = mock_run.call_args_list[1]
        args = text_call[0][0]
        assert "-l" in args
        assert "Needs more detail on testing approach" in args

    async def test_empty_reason(self, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        result = await reject_plan("leo", "")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_not_in_plan_waiting(self, state_dir):
        _write_state(state_dir, "leo", {"state": "idle"})
        result = await reject_plan("leo", "bad plan")
        assert result["success"] is False
        assert "not in plan_waiting" in result["error"]

    async def test_invalid_session_name(self, state_dir):
        result = await reject_plan("-invalid", "reason")
        assert result["success"] is False

    @patch(f"{MODULE}._run_command")
    async def test_session_does_not_exist(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        mock_run.return_value = (1, "", "session not found")

        result = await reject_plan("leo", "reason")
        assert result["success"] is False
        assert "does not exist" in result["error"]

    @patch(f"{MODULE}._run_command")
    async def test_send_text_failure(self, mock_run, state_dir):
        _write_state(state_dir, "leo", {"state": "plan_waiting"})
        mock_run.side_effect = [
            (0, "", ""),  # has-session
            (1, "", "send failed"),  # send-keys -l
        ]

        result = await reject_plan("leo", "reason")
        assert result["success"] is False
        assert "Failed to send" in result["error"]


# ---------------------------------------------------------------------------
# TestRegisterPlanTools
# ---------------------------------------------------------------------------


class TestRegisterPlanTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_approvals_tools(registry, StateFileApprovals())

        names = registry.get_tool_names()
        assert "list_agent_plans" in names
        assert "approve_plan" in names
        assert "reject_plan" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_approvals_tools(registry, StateFileApprovals())
        assert len(registry._backend_definitions) == 3

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_approvals_tools(registry, StateFileApprovals())
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_approvals_tools(registry, StateFileApprovals())
        for name in ["list_agent_plans", "approve_plan", "reject_plan"]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
