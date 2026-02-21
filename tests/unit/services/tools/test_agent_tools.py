"""Tests for agent management tools."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lovely_assistant.services.tools._agent_tools import (
    AGENT_REGISTRY,
    MAX_SESSION_NAME_LENGTH,
    _read_state_file,
    _run_command,
    _validate_session_name,
    _validate_working_directory,
    check_agent_state,
    list_agents,
    register_agent_tools,
    send_agent_message,
    start_agent,
    stop_agent,
)
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.config import ToolConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULE = "lovely_assistant.services.tools._agent_tools"


def _mock_run(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Create an AsyncMock for _run_command with given outputs."""
    return AsyncMock(return_value=(returncode, stdout, stderr))


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR to a temp directory."""
    monkeypatch.setattr(f"{MODULE}.STATE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def workspace_dir(tmp_path, monkeypatch):
    """Redirect WORKSPACE_ROOT to a temp directory with a sub-dir."""
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(f"{MODULE}.WORKSPACE_ROOT", ws)
    return ws


# ---------------------------------------------------------------------------
# TestRunCommand
# ---------------------------------------------------------------------------


class TestRunCommand:
    async def test_successful_command(self):
        rc, stdout, stderr = await _run_command(["echo", "hello"])
        assert rc == 0
        assert stdout == "hello"

    async def test_command_not_found(self):
        rc, stdout, stderr = await _run_command(["nonexistent_command_xyz"])
        assert rc == 127
        assert "Command not found" in stderr

    async def test_nonzero_exit(self):
        rc, stdout, stderr = await _run_command(["false"])
        assert rc != 0


# ---------------------------------------------------------------------------
# TestReadStateFile
# ---------------------------------------------------------------------------


class TestReadStateFile:
    def test_valid_json(self, state_dir):
        state_file = state_dir / "leo.json"
        state_file.write_text(
            json.dumps(
                {
                    "entity": "leo",
                    "state": "idle",
                    "issue": None,
                }
            )
        )
        result = _read_state_file("leo")
        assert result is not None
        assert result["entity"] == "leo"
        assert result["state"] == "idle"

    def test_missing_file(self, state_dir):
        result = _read_state_file("nonexistent")
        assert result is None

    def test_invalid_json(self, state_dir):
        state_file = state_dir / "broken.json"
        state_file.write_text("not valid json {{{")
        result = _read_state_file("broken")
        assert result is None

    def test_non_dict_json(self, state_dir):
        state_file = state_dir / "array.json"
        state_file.write_text(json.dumps([1, 2, 3]))
        result = _read_state_file("array")
        assert result is None


# ---------------------------------------------------------------------------
# TestValidateSessionName
# ---------------------------------------------------------------------------


class TestValidateSessionName:
    @pytest.mark.parametrize(
        "name",
        [
            "leo",
            "agent-backbone",
            "platform-api",
            "test123",
            "a",
            "A1-b2-c3",
        ],
    )
    def test_valid_names(self, name):
        assert _validate_session_name(name) is None

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("", "empty"),
            ("-starts-with-dash", "starts with dash"),
            ("has spaces", "has spaces"),
            ("has.dots", "has dots"),
            ("has_underscores", "has underscores"),
            ("has/slash", "has slash"),
            ("a" * (MAX_SESSION_NAME_LENGTH + 1), "too long"),
        ],
    )
    def test_invalid_names(self, name, reason):
        result = _validate_session_name(name)
        assert result is not None, f"Expected invalid for: {reason}"


# ---------------------------------------------------------------------------
# TestValidateWorkingDirectory
# ---------------------------------------------------------------------------


class TestValidateWorkingDirectory:
    def test_valid_path(self, workspace_dir):
        subdir = workspace_dir / "core" / "code"
        subdir.mkdir(parents=True)
        result = _validate_working_directory(str(subdir))
        assert result is None

    def test_outside_workspace(self, workspace_dir):
        result = _validate_working_directory("/tmp/not-in-workspace")
        assert result is not None
        assert "must be under" in result

    def test_nonexistent_directory(self, workspace_dir):
        result = _validate_working_directory(str(workspace_dir / "does-not-exist"))
        assert result is not None
        assert "does not exist" in result


# ---------------------------------------------------------------------------
# TestAgentRegistry
# ---------------------------------------------------------------------------


class TestAgentRegistry:
    def test_registry_has_expected_agents(self):
        expected = {
            "bell",
            "feynman",
            "ike",
            "leo",
            "hamilton",
            "curie",
            "ada",
            "brunel",
            "agent-backbone",
            "agent-orchestration-dashboard",
            "lovely-assistant",
            "alfred",
        }
        assert set(AGENT_REGISTRY.keys()) == expected

    def test_registry_paths_are_absolute(self):
        for name, path in AGENT_REGISTRY.items():
            assert path.is_absolute(), f"{name} has relative path: {path}"

    def test_registry_paths_under_home(self):
        home = Path.home()
        for name, path in AGENT_REGISTRY.items():
            assert str(path).startswith(str(home)), f"{name} not under home: {path}"


# ---------------------------------------------------------------------------
# TestListAgents
# ---------------------------------------------------------------------------


class TestListAgents:
    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_with_sessions_and_state(self, mock_state, mock_run):
        mock_run.return_value = (0, "leo\nike\nada", "")
        mock_state.side_effect = [
            {"entity": "leo", "state": "idle", "issue": None, "context": None},
            {"entity": "ike", "state": "processing", "issue": 42, "context": "Working on #42"},
            None,
        ]

        result = await list_agents()
        assert result["success"] is True
        assert result["count"] == 3
        assert len(result["sessions"]) == 3
        assert result["sessions"][0]["state"] == "idle"
        assert result["sessions"][1]["state"] == "processing"
        assert result["sessions"][1]["issue"] == 42
        assert result["sessions"][2]["state"] == "unknown"

    @patch(f"{MODULE}._run_command")
    async def test_empty_sessions(self, mock_run):
        mock_run.return_value = (1, "", "no server running")

        result = await list_agents()
        assert result["success"] is True
        assert result["sessions"] == []

    @patch(f"{MODULE}._run_command")
    async def test_tmux_not_installed(self, mock_run):
        mock_run.return_value = (127, "", "Command not found: tmux")

        result = await list_agents()
        assert result["success"] is False
        assert "not installed" in result["error"]

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_missing_state_files(self, mock_state, mock_run):
        mock_run.return_value = (0, "session1\nsession2", "")
        mock_state.return_value = None

        result = await list_agents()
        assert result["success"] is True
        assert all(s["state"] == "unknown" for s in result["sessions"])

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_includes_registered_directory(self, mock_state, mock_run):
        mock_run.return_value = (0, "leo\nunknown-session", "")
        mock_state.return_value = None

        result = await list_agents()
        assert result["success"] is True
        sessions_by_name = {s["session_name"]: s for s in result["sessions"]}
        # leo is in registry
        assert sessions_by_name["leo"]["registered_directory"] is not None
        # unknown-session is not
        assert sessions_by_name["unknown-session"]["registered_directory"] is None


# ---------------------------------------------------------------------------
# TestCheckAgentState
# ---------------------------------------------------------------------------


class TestCheckAgentState:
    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_exists_with_state(self, mock_state, mock_run):
        mock_state.return_value = {"entity": "leo", "state": "idle", "issue": None}
        # has-session check, then capture-pane
        mock_run.side_effect = [
            (0, "", ""),
            (0, "$ claude\nHello!", ""),
        ]

        result = await check_agent_state("leo")
        assert result["success"] is True
        assert result["session_exists"] is True
        assert result["state"]["entity"] == "leo"
        assert "recent_output" in result

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_exists_without_state(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]

        result = await check_agent_state("leo")
        assert result["success"] is True
        assert result["state"] is None
        assert result["note"] == "No state file found"

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_session_does_not_exist(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.return_value = (1, "", "session not found")

        result = await check_agent_state("nonexistent")
        assert result["success"] is True
        assert result["session_exists"] is False
        assert "recent_output" not in result

    async def test_invalid_name(self):
        result = await check_agent_state("-invalid")
        assert result["success"] is False
        assert "error" in result

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_recent_output_captured(self, mock_state, mock_run):
        mock_state.return_value = {"state": "idle"}
        mock_run.side_effect = [
            (0, "", ""),
            (0, "line1\nline2\nline3", ""),
        ]

        result = await check_agent_state("leo")
        assert result["recent_output"] == "line1\nline2\nline3"


# ---------------------------------------------------------------------------
# TestStartAgent
# ---------------------------------------------------------------------------


class TestStartAgent:
    @patch(f"{MODULE}._run_command")
    async def test_success(self, mock_run, workspace_dir):
        target = workspace_dir / "core" / "code"
        target.mkdir(parents=True)

        # has-session (not found), new-session, send-keys (claude)
        mock_run.side_effect = [
            (1, "", "session not found"),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await start_agent("test-agent", str(target))
        assert result["success"] is True
        assert result["session_name"] == "test-agent"
        assert result["working_directory"] == str(target)

    @patch(f"{MODULE}._run_command")
    async def test_existing_session_rejected(self, mock_run, workspace_dir):
        target = workspace_dir / "repo"
        target.mkdir()
        mock_run.return_value = (0, "", "")  # has-session succeeds

        result = await start_agent("existing", str(target))
        assert result["success"] is False
        assert "already exists" in result["error"]

    async def test_invalid_name(self, workspace_dir):
        target = workspace_dir / "repo"
        target.mkdir()
        result = await start_agent("-bad-name", str(target))
        assert result["success"] is False

    async def test_path_outside_workspace(self, workspace_dir):
        result = await start_agent("test-agent", "/tmp/outside")
        assert result["success"] is False
        assert "must be under" in result["error"]

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock)
    async def test_with_initial_prompt(self, mock_sleep, mock_run, workspace_dir):
        target = workspace_dir / "repo"
        target.mkdir()

        # has-session, new-session, send-keys(claude), send-keys(prompt), send-keys(Enter)
        mock_run.side_effect = [
            (1, "", ""),
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await start_agent("test-agent", str(target), "do the thing")
        assert result["success"] is True
        assert result["initial_prompt"] == "do the thing"
        mock_sleep.assert_awaited_once_with(2)

    @patch(f"{MODULE}._run_command")
    async def test_success_with_registry_fallback(self, mock_run, workspace_dir):
        target = workspace_dir / "core" / "bell"
        target.mkdir(parents=True)
        # Patch registry to use the temp workspace path
        with patch(f"{MODULE}.AGENT_REGISTRY", {"bell": target}):
            # has-session (not found), new-session, send-keys (claude)
            mock_run.side_effect = [
                (1, "", "session not found"),
                (0, "", ""),
                (0, "", ""),
            ]

            result = await start_agent("bell")
            assert result["success"] is True
            assert result["working_directory"] == str(target)

    @patch(f"{MODULE}._run_command")
    async def test_registry_override_by_explicit_directory(self, mock_run, workspace_dir):
        explicit = workspace_dir / "explicit" / "path"
        explicit.mkdir(parents=True)

        mock_run.side_effect = [
            (1, "", "session not found"),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await start_agent("bell", str(explicit))
        assert result["success"] is True
        assert result["working_directory"] == str(explicit)

    async def test_unknown_agent_without_directory_fails(self):
        result = await start_agent("unknown-agent-xyz")
        assert result["success"] is False
        assert "not in the agent registry" in result["error"]

    @patch(f"{MODULE}._run_command")
    async def test_registry_path_not_exist_fails(self, mock_run, workspace_dir):
        nonexistent = workspace_dir / "does" / "not" / "exist"
        with patch(f"{MODULE}.AGENT_REGISTRY", {"test-agent": nonexistent}):
            result = await start_agent("test-agent")
            assert result["success"] is False
            assert "does not exist" in result["error"]

    @patch(f"{MODULE}._run_command")
    async def test_registry_path_outside_workspace_succeeds(self, mock_run, tmp_path):
        """Registry paths outside ~/ws/ (e.g. feynman, brunel) should bypass workspace check."""
        outside_ws = tmp_path / "orchestration"
        outside_ws.mkdir()
        with patch(f"{MODULE}.AGENT_REGISTRY", {"feynman": outside_ws}):
            mock_run.side_effect = [
                (1, "", "session not found"),
                (0, "", ""),
                (0, "", ""),
            ]
            result = await start_agent("feynman")
            assert result["success"] is True
            assert result["working_directory"] == str(outside_ws)


# ---------------------------------------------------------------------------
# TestStopAgent
# ---------------------------------------------------------------------------


class TestStopAgent:
    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_success(self, mock_state, mock_run):
        mock_state.return_value = {"state": "idle"}
        # has-session, kill-session
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]

        result = await stop_agent("leo")
        assert result["success"] is True
        assert result["session_name"] == "leo"
        assert result["previous_state"] == "idle"

    @patch(f"{MODULE}._run_command")
    async def test_nonexistent_session(self, mock_run):
        mock_run.return_value = (1, "", "session not found")

        result = await stop_agent("ghost")
        assert result["success"] is False
        assert "does not exist" in result["error"]

    async def test_invalid_name(self):
        result = await stop_agent("-bad-name")
        assert result["success"] is False
        assert "error" in result

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_kill_failure(self, mock_state, mock_run):
        mock_state.return_value = {"state": "processing"}
        # has-session succeeds, kill-session fails
        mock_run.side_effect = [
            (0, "", ""),
            (1, "", "cannot kill session"),
        ]

        result = await stop_agent("stuck")
        assert result["success"] is False
        assert "Failed to kill session" in result["error"]

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_no_state_file(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]

        result = await stop_agent("no-state")
        assert result["success"] is True
        assert result["previous_state"] == "unknown"

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_captures_processing_state(self, mock_state, mock_run):
        mock_state.return_value = {"state": "processing", "issue": 42}
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
        ]

        result = await stop_agent("busy-agent")
        assert result["success"] is True
        assert result["previous_state"] == "processing"


# ---------------------------------------------------------------------------
# TestSendAgentMessage
# ---------------------------------------------------------------------------


class TestSendAgentMessage:
    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_success(self, mock_state, mock_run):
        mock_state.return_value = {"state": "idle"}
        # has-session, send-keys -l, send-keys Enter
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await send_agent_message("leo", "check status")
        assert result["success"] is True
        assert result["message_sent"] == "check status"
        assert result["agent_state"] == "idle"

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_envelope_tag_sent(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        await send_agent_message("leo", "hello")

        # The second call is send-keys -l with the envelope
        send_call = mock_run.call_args_list[1]
        args = send_call[0][0]  # positional args to _run_command
        assert "[via:lovely-assistant from:elias]" in args[-1]
        assert "hello" in args[-1]

    @patch(f"{MODULE}._run_command")
    async def test_nonexistent_session(self, mock_run):
        mock_run.return_value = (1, "", "session not found")

        result = await send_agent_message("ghost", "hello")
        assert result["success"] is False
        assert "does not exist" in result["error"]

    async def test_empty_message(self):
        result = await send_agent_message("leo", "")
        assert result["success"] is False
        assert "empty" in result["error"]

    async def test_whitespace_only_message(self):
        result = await send_agent_message("leo", "   ")
        assert result["success"] is False
        assert "empty" in result["error"]

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_busy_agent_warning(self, mock_state, mock_run):
        mock_state.return_value = {"state": "processing"}
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        result = await send_agent_message("leo", "urgent question")
        assert result["success"] is True
        assert "warning" in result
        assert "processing" in result["warning"]

    async def test_invalid_session_name(self):
        result = await send_agent_message("-bad", "hello")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestRegisterAgentTools
# ---------------------------------------------------------------------------


class TestRegisterAgentTools:
    def test_all_tools_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)

        names = registry.get_tool_names()
        assert "list_agents" in names
        assert "check_agent_state" in names
        assert "start_agent" in names
        assert "stop_agent" in names
        assert "send_agent_message" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)
        assert len(registry._backend_definitions) == 5

    def test_all_are_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_definitions_have_schemas(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)
        for defn in registry._backend_definitions.values():
            assert isinstance(defn.parameters_schema, dict)
            assert defn.description

    def test_handlers_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)
        for name in [
            "list_agents",
            "check_agent_state",
            "start_agent",
            "stop_agent",
            "send_agent_message",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])

    def test_start_agent_working_directory_not_required(self):
        registry = ToolRegistry(ToolConfig())
        register_agent_tools(registry)
        schema = registry._backend_definitions["start_agent"].parameters_schema
        required = schema.get("required", [])
        assert "working_directory" not in required
        assert "session_name" in required
