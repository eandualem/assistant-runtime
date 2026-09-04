"""Tests for agent management tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools._request_context import assistant_request_context
from assistant_runtime.services.tools.capabilities.peers import register_peers_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.backbone.peers import (
    MAX_SESSION_NAME_LENGTH,
    BackbonePeers,
    _read_state_file,
    _run_command,
    _validate_session_name,
    check_agent_state,
    get_active_agents,
    list_agents,
    send_agent_message,
    start_agent,
    stop_agent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODULE = "assistant_runtime.services.tools.providers.backbone.peers"


def _mock_run(returncode: int = 0, stdout: str = "", stderr: str = ""):
    """Create an AsyncMock for _run_command with given outputs."""
    return AsyncMock(return_value=(returncode, stdout, stderr))


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect STATE_DIR to a temp directory."""
    monkeypatch.setattr(f"{MODULE}.STATE_DIR", tmp_path)
    return tmp_path


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

    def test_invalid_json_logs_warning(self, state_dir):
        state_file = state_dir / "broken.json"
        state_file.write_text("not valid json {{{")
        with patch(f"{MODULE}.logger") as mock_logger:
            result = _read_state_file("broken")
        assert result is None
        mock_logger.warning.assert_called_once()
        assert mock_logger.warning.call_args[0][0] == "Corrupted agent state file"

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

        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(return_value=[])
        mock_cache.get_agent_info = MagicMock(return_value=None)
        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
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

        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(return_value=[])
        mock_cache.get_agent_info = MagicMock(return_value=None)
        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
            result = await list_agents()
        assert result["success"] is True
        assert all(s["state"] == "unknown" for s in result["sessions"])

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_enriches_from_registry_cache(self, mock_state, mock_run):
        mock_run.return_value = (0, "leo\nunknown-session", "")
        mock_state.return_value = None

        leo_info = {
            "display_name": "Leo",
            "role": "Strategy Co-Architect",
            "type": "orchestrator",
            "home": "/srv/agents/leo",
        }
        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(return_value=[{"session": "leo", **leo_info}])
        mock_cache.get_agent_info = MagicMock(
            side_effect=lambda name: leo_info if name == "leo" else None
        )

        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
            result = await list_agents()

        assert result["success"] is True
        sessions_by_name = {s["session_name"]: s for s in result["sessions"]}
        # leo is in registry cache — enriched with display_name, role, type, home
        assert sessions_by_name["leo"]["display_name"] == "Leo"
        assert sessions_by_name["leo"]["role"] == "Strategy Co-Architect"
        assert sessions_by_name["leo"]["type"] == "orchestrator"
        assert sessions_by_name["leo"]["home"] == "/srv/agents/leo"
        # unknown-session is NOT in cache — no enrichment fields
        assert "display_name" not in sessions_by_name["unknown-session"]
        assert "role" not in sessions_by_name["unknown-session"]
        assert "type" not in sessions_by_name["unknown-session"]
        assert "home" not in sessions_by_name["unknown-session"]


# ---------------------------------------------------------------------------
# TestGetActiveAgents
# ---------------------------------------------------------------------------


class TestGetActiveAgents:
    async def test_returns_only_online_real_agents(self):
        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(
            return_value=[
                {
                    "session": "ike",
                    "display_name": "Eisenhower",
                    "role": "Core Orchestrator",
                    "type": "named_entity",
                    "state": "idle",
                    "runtime": "codex",
                    "online": True,
                    "current_issue": None,
                },
                {
                    "session": "gateway",
                    "display_name": "gateway",
                    "role": "infra",
                    "type": "service",
                    "state": "unknown",
                    "runtime": None,
                    "online": True,
                    "current_issue": None,
                },
                {
                    "session": "leo",
                    "display_name": "Vinci",
                    "role": "Strategy Co-Architect",
                    "type": "named_entity",
                    "state": "offline",
                    "runtime": None,
                    "online": False,
                    "current_issue": None,
                },
                {
                    "session": "mystery-session",
                    "display_name": "mystery-session",
                    "role": "",
                    "type": "coding_agent",
                    "state": "unknown",
                    "runtime": None,
                    "online": True,
                    "current_issue": None,
                },
            ]
        )

        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
            result = await get_active_agents()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["agents"] == [
            {
                "session_name": "ike",
                "display_name": "Eisenhower",
                "role": "Core Orchestrator",
                "type": "named_entity",
                "state": "idle",
                "runtime": "codex",
                "current_issue": None,
            }
        ]

    async def test_keeps_unknown_state_when_runtime_present(self):
        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(
            return_value=[
                {
                    "session": "bell-wf",
                    "display_name": "Bell",
                    "role": "Org Orchestrator",
                    "type": "named_entity",
                    "state": "unknown",
                    "runtime": "claude",
                    "online": True,
                    "current_issue": 813,
                }
            ]
        )

        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
            result = await get_active_agents()

        assert result["success"] is True
        assert result["count"] == 1
        assert result["agents"][0]["session_name"] == "bell-wf"
        assert result["agents"][0]["state"] == "unknown"
        assert result["agents"][0]["runtime"] == "claude"

    async def test_returns_error_when_backbone_unavailable(self):
        mock_cache = MagicMock()
        mock_cache.get_agents = AsyncMock(return_value=None)

        with patch(f"{MODULE}.get_registry_cache", return_value=mock_cache):
            result = await get_active_agents()

        assert result == {"error": "Backbone agent registry unavailable", "success": False}


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
    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success_basic(self, mock_backbone):
        mock_backbone.return_value = (
            200,
            {"working_directory": "/srv/agents/leo", "session": "leo"},
        )

        result = await start_agent("leo")
        assert result["success"] is True
        assert result["session_name"] == "leo"
        assert result["runtime"] == "claude"
        assert result["model"] is None
        assert result["resume"] is False
        assert result["working_directory"] == "/srv/agents/leo"
        assert result["initial_prompt"] is None

        # Verify backbone was called with correct args
        mock_backbone.assert_awaited_once_with(
            "POST",
            "/api/agents/leo/start",
            json_body={"runtime": "claude"},
        )

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success_with_runtime_and_model(self, mock_backbone):
        mock_backbone.return_value = (
            200,
            {"working_directory": "/srv/agents/leo"},
        )

        result = await start_agent("leo", runtime="aider", model="sonnet")
        assert result["success"] is True
        assert result["runtime"] == "aider"
        assert result["model"] == "sonnet"

        mock_backbone.assert_awaited_once_with(
            "POST",
            "/api/agents/leo/start",
            json_body={"runtime": "aider", "model": "sonnet"},
        )

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success_with_resume(self, mock_backbone):
        mock_backbone.return_value = (
            200,
            {"working_directory": "/srv/agents/leo"},
        )

        result = await start_agent("leo", resume=True)
        assert result["success"] is True
        assert result["resume"] is True

        mock_backbone.assert_awaited_once_with(
            "POST",
            "/api/agents/leo/start",
            json_body={"runtime": "claude", "resume": True},
        )

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success_with_initial_prompt(self, mock_backbone, mock_run):
        mock_backbone.return_value = (
            200,
            {"working_directory": "/srv/agents/leo"},
        )
        mock_run.side_effect = [
            (0, "", ""),  # send-keys prompt
            (0, "", ""),  # send-keys Enter
        ]

        with patch(f"{MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await start_agent("leo", initial_prompt="do the thing")
            assert result["success"] is True
            assert result["initial_prompt"] == "do the thing"
            mock_sleep.assert_awaited_once_with(2)

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_backbone_unavailable(self, mock_backbone):
        mock_backbone.return_value = (
            -1,
            {"error": "Request timed out: POST /api/agents/leo/start", "success": False},
        )

        result = await start_agent("leo")
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_backbone_400_unknown_runtime(self, mock_backbone):
        mock_backbone.return_value = (
            400,
            {"error": "Unknown runtime 'zsh'. Available: claude, aider, gemini"},
        )

        result = await start_agent("leo", runtime="zsh")
        assert result["success"] is False
        assert "Unknown runtime" in result["error"]

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_backbone_404_unknown_session(self, mock_backbone):
        mock_backbone.return_value = (
            404,
            {"error": "Agent 'nonexistent' not found in registry"},
        )

        result = await start_agent("nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_invalid_session_name(self):
        result = await start_agent("-bad-name")
        assert result["success"] is False


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

        with assistant_request_context("sess_abc123"):
            result = await send_agent_message("leo", "check status")
        assert result["success"] is True
        assert result["message_sent"] == "check status"
        assert result["agent_state"] == "idle"
        assert result["reply_session_id"] == "sess_abc123"

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_envelope_tag_sent(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.side_effect = [
            (0, "", ""),
            (0, "", ""),
            (0, "", ""),
        ]

        with assistant_request_context("sess_abc123"):
            await send_agent_message("leo", "hello")

        # The second call is send-keys -l with the envelope
        send_call = mock_run.call_args_list[1]
        args = send_call[0][0]  # positional args to _run_command
        assert "[via:assistant from:operator session:sess_abc123]" in args[-1]
        assert "hello" in args[-1]

    @patch(f"{MODULE}._run_command")
    @patch(f"{MODULE}._read_state_file")
    async def test_missing_request_context_rejects_send(self, mock_state, mock_run):
        mock_state.return_value = None
        mock_run.return_value = (0, "", "")

        result = await send_agent_message("leo", "hello")

        assert result["success"] is False
        assert "assistant session context" in result["error"]
        assert mock_run.await_count == 1

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

        with assistant_request_context("sess_busy"):
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
        register_peers_tools(registry, BackbonePeers())

        names = registry.get_tool_names()
        assert "list_agents" in names
        assert "get_active_agents" in names
        assert "check_agent_state" in names
        assert "start_agent" in names
        assert "stop_agent" in names
        assert "send_agent_message" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_peers_tools(registry, BackbonePeers())
        assert len(registry._backend_definitions) == 6

    def test_all_are_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_peers_tools(registry, BackbonePeers())
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_definitions_have_schemas(self):
        registry = ToolRegistry(ToolConfig())
        register_peers_tools(registry, BackbonePeers())
        for defn in registry._backend_definitions.values():
            assert isinstance(defn.parameters_schema, dict)
            assert defn.description

    def test_handlers_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_peers_tools(registry, BackbonePeers())
        for name in [
            "list_agents",
            "get_active_agents",
            "check_agent_state",
            "start_agent",
            "stop_agent",
            "send_agent_message",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])

    def test_start_agent_schema_has_new_params(self):
        registry = ToolRegistry(ToolConfig())
        register_peers_tools(registry, BackbonePeers())
        schema = registry._backend_definitions["start_agent"].parameters_schema
        properties = schema.get("properties", {})
        assert "working_directory" not in properties
        assert "session_name" in schema.get("required", [])
        assert "runtime" in properties
        assert "model" in properties
        assert "resume" in properties
        assert "initial_prompt" in properties
