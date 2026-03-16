"""Agent management tools -- list, inspect, start, and message AI agents."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from lovely_assistant.services.tools._agent_registry_cache import get_registry_cache
from lovely_assistant.services.tools._backbone_client import backbone_error, backbone_request
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._request_context import get_current_assistant_session_id
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

STATE_DIR = Path.home() / ".claude" / "state"
SESSION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$")
MAX_SESSION_NAME_LENGTH = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _run_command(args: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            (stdout_bytes or b"").decode().strip(),
            (stderr_bytes or b"").decode().strip(),
        )
    except FileNotFoundError:
        return (127, "", f"Command not found: {args[0]}")
    except TimeoutError:
        proc.kill()
        return (1, "", f"Command timed out after {timeout}s")


def _read_state_file(session_name: str) -> dict[str, Any] | None:
    """Read an agent state file. Returns None if missing or invalid."""
    state_path = STATE_DIR / f"{session_name}.json"
    try:
        data = json.loads(state_path.read_text())
        if isinstance(data, dict):
            return data
        return None
    except FileNotFoundError:
        return None  # Agent has no state file — normal
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Corrupted agent state file", path=str(state_path), error=str(e))
        return None


def _validate_session_name(session_name: str) -> str | None:
    """Validate a tmux session name. Returns error message or None if valid."""
    if not session_name:
        return "Session name cannot be empty"
    if len(session_name) > MAX_SESSION_NAME_LENGTH:
        return f"Session name too long (max {MAX_SESSION_NAME_LENGTH} chars)"
    if not SESSION_NAME_PATTERN.match(session_name):
        return "Session name must be alphanumeric with hyphens, starting with alphanumeric"
    return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def list_agents() -> dict[str, Any]:
    """List all running tmux sessions with their agent state."""
    rc, stdout, stderr = await _run_command(["tmux", "list-sessions", "-F", "#{session_name}"])
    if rc == 127:
        return {"error": "tmux is not installed", "success": False}
    if rc != 0:
        # tmux returns error when no sessions exist
        return {"sessions": [], "success": True}

    cache = get_registry_cache()
    await cache.get_agents()  # ensure cache is populated

    session_names = [s for s in stdout.splitlines() if s.strip()]
    sessions = []
    for name in session_names:
        state = _read_state_file(name)
        info = cache.get_agent_info(name)
        entry: dict[str, Any] = {
            "session_name": name,
            "state": state.get("state", "unknown") if state else "unknown",
            "entity": state.get("entity") if state else None,
            "issue": state.get("issue") if state else None,
            "context": state.get("context") if state else None,
        }
        if info:
            entry["display_name"] = info.get("display_name")
            entry["role"] = info.get("role")
            entry["type"] = info.get("type")
            entry["home"] = info.get("home")
            entry["runtime"] = info.get("runtime")
        sessions.append(entry)

    return {"sessions": sessions, "count": len(sessions), "success": True}


# Infrastructure sessions to exclude from get_active_agents.
_INFRASTRUCTURE_SESSIONS = frozenset(
    {
        "gateway",
        "prefect",
        "telegram-bot",
        "backbone-worker",
        "ngrok",
    }
)


async def get_active_agents() -> dict[str, Any]:
    """List active AI agents, excluding infrastructure and offline sessions.

    Uses the backbone registry API which provides runtime, state, and
    entity metadata. Filters out infrastructure sessions, unknown state,
    and offline agents.
    """
    cache = get_registry_cache()
    agents = await cache.get_agents()
    if agents is None:
        return {"error": "Backbone agent registry unavailable", "success": False}

    active = []
    for agent in agents:
        session = agent.get("session", "")
        # Skip infrastructure
        if session in _INFRASTRUCTURE_SESSIONS:
            continue
        # Skip offline
        if not agent.get("online"):
            continue
        # Skip unknown state (no state file, not a real agent session)
        state = agent.get("state", "unknown")
        if state == "unknown" and agent.get("runtime") is None:
            continue

        active.append(
            {
                "session_name": session,
                "display_name": agent.get("display_name") or session,
                "role": agent.get("role", ""),
                "type": agent.get("type", ""),
                "state": state,
                "runtime": agent.get("runtime"),
                "current_issue": agent.get("current_issue"),
            }
        )

    return {"agents": active, "count": len(active), "success": True}


async def check_agent_state(session_name: str) -> dict[str, Any]:
    """Check detailed state of a specific agent session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    state = _read_state_file(session_name)

    # Check if the tmux session actually exists
    rc, stdout, stderr = await _run_command(["tmux", "has-session", "-t", session_name])
    session_exists = rc == 0

    result: dict[str, Any] = {
        "session_name": session_name,
        "session_exists": session_exists,
        "success": True,
    }

    if state:
        result["state"] = state
    else:
        result["state"] = None
        result["note"] = "No state file found"

    # Capture recent output if session exists
    if session_exists:
        rc, output, _ = await _run_command(
            ["tmux", "capture-pane", "-t", session_name, "-p", "-S", "-10"]
        )
        if rc == 0 and output:
            result["recent_output"] = output

    return result


async def start_agent(
    session_name: str,
    runtime: str = "claude",
    model: str | None = None,
    resume: bool = False,
    initial_prompt: str = "",
) -> dict[str, Any]:
    """Start a new agent in a tmux session via the backbone start endpoint.

    The backbone handles working directory resolution, command construction,
    and tmux session creation. After the session starts, an optional initial
    prompt is sent directly via tmux send-keys.
    """
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    # Delegate to backbone start endpoint
    body: dict[str, Any] = {"runtime": runtime}
    if model is not None:
        body["model"] = model
    if resume:
        body["resume"] = True

    status, data = await backbone_request(
        "POST",
        f"/api/agents/{session_name}/start",
        json_body=body,
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status != 200:
        return {"error": backbone_error(data), "success": False}

    working_directory = data.get("working_directory", "")

    # Send initial prompt if provided (with delay to let the CLI start)
    if initial_prompt:
        await asyncio.sleep(2)
        rc, _, stderr = await _run_command(
            ["tmux", "send-keys", "-t", session_name, "-l", initial_prompt]
        )
        if rc == 0:
            await _run_command(["tmux", "send-keys", "-t", session_name, "Enter"])

    logger.info(
        "Started agent session",
        session=session_name,
        runtime=runtime,
        directory=working_directory,
    )
    return {
        "session_name": session_name,
        "runtime": runtime,
        "model": model,
        "resume": resume,
        "working_directory": working_directory,
        "initial_prompt": initial_prompt or None,
        "success": True,
    }


async def stop_agent(session_name: str) -> dict[str, Any]:
    """Stop an agent by killing its tmux session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    # Check if session exists
    rc, _, _ = await _run_command(["tmux", "has-session", "-t", session_name])
    if rc != 0:
        return {
            "error": f"Session '{session_name}' does not exist",
            "success": False,
        }

    # Read state before killing for response context
    state = _read_state_file(session_name)
    previous_state = state.get("state", "unknown") if state else "unknown"

    # Kill the session
    rc, _, stderr = await _run_command(["tmux", "kill-session", "-t", session_name])
    if rc != 0:
        return {"error": f"Failed to kill session: {stderr}", "success": False}

    logger.info("Stopped agent session", session=session_name, previous_state=previous_state)
    return {
        "session_name": session_name,
        "previous_state": previous_state,
        "success": True,
    }


async def send_agent_message(session_name: str, message: str) -> dict[str, Any]:
    """Send a message to a running agent session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    if not message or not message.strip():
        return {"error": "Message cannot be empty", "success": False}

    # Verify session exists
    rc, _, _ = await _run_command(["tmux", "has-session", "-t", session_name])
    if rc != 0:
        return {"error": f"Session '{session_name}' does not exist", "success": False}

    # Check current state for context
    state = _read_state_file(session_name)
    state_warning = None
    if state and state.get("state") in ("processing", "busy"):
        state_warning = f"Agent is currently {state['state']} — message will be queued"

    reply_session_id = get_current_assistant_session_id()
    if not reply_session_id:
        return {
            "error": "Cannot send Jarvis-routed message without assistant session context",
            "success": False,
        }

    # Send with reply-safe Jarvis envelope
    envelope = f"[via:jarvis from:elias session:{reply_session_id}] {message.strip()}"
    rc, _, stderr = await _run_command(["tmux", "send-keys", "-t", session_name, "-l", envelope])
    if rc != 0:
        return {"error": f"Failed to send message: {stderr}", "success": False}

    # Press Enter to submit
    rc, _, stderr = await _run_command(["tmux", "send-keys", "-t", session_name, "Enter"])
    if rc != 0:
        return {"error": f"Failed to send Enter: {stderr}", "success": False}

    result: dict[str, Any] = {
        "session_name": session_name,
        "message_sent": message.strip(),
        "reply_session_id": reply_session_id,
        "success": True,
    }
    if state_warning:
        result["warning"] = state_warning
    if state:
        result["agent_state"] = state.get("state", "unknown")

    return result


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_agent_tools(registry: ToolRegistry) -> None:
    """Register all agent management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="list_agents",
            description=(
                "List ALL tmux sessions, including infrastructure and support services. "
                "Returns session names, state, entity, runtime, role, type, and home "
                "directory. This is a low-level tmux view. Prefer get_active_agents for "
                "normal 'agent status' requests."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        list_agents,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_active_agents",
            description=(
                "List active AI agents — the PRIMARY tool for 'agent status', "
                "'what agents are running', or similar requests. Returns only real "
                "agents (named entities + coding agents) that are online, excluding "
                "infrastructure sessions like gateway, prefect, ngrok, and workers. "
                "Use this instead of list_agents unless the user explicitly wants the "
                "full tmux session list. Each agent includes session_name, display_name, "
                "role, type, state, runtime, and current_issue."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        get_active_agents,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="check_agent_state",
            description=(
                "Check the detailed state of a specific agent session, including "
                "state file data and recent terminal output."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session to check",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        check_agent_state,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="start_agent",
            description=(
                "Start a new AI agent in a tmux session. Delegates to the backbone "
                "start endpoint which handles working directory resolution and session "
                "creation. Supports runtime selection, model override, and resume mode. "
                "Optionally sends an initial prompt after the CLI starts. The result of "
                "this tool is authoritative for whether the start succeeded, so do not "
                "navigate or call look_at_screen just to confirm unless the user "
                "explicitly asks for UI verification. Use get_active_agents to inspect "
                "current agent status and known session names."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name for the new tmux session (must be a known agent)",
                    },
                    "runtime": {
                        "type": "string",
                        "description": (
                            "AI CLI runtime to use (for example claude, codex, gemini, "
                            "cursor, opencode, shell)"
                        ),
                        "default": "claude",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model override for the runtime (e.g. opus, sonnet). Optional — uses runtime default if omitted.",
                    },
                    "resume": {
                        "type": "boolean",
                        "description": "Resume the most recent conversation instead of starting fresh",
                        "default": False,
                    },
                    "initial_prompt": {
                        "type": "string",
                        "description": "Optional prompt to send after the CLI starts",
                        "default": "",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        start_agent,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="stop_agent",
            description=(
                "Stop a running AI agent by killing its tmux session. "
                "Returns the agent's previous state before termination."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session to stop",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        stop_agent,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="send_agent_message",
            description=(
                "Send a message to a running agent session. Prepends the "
                "[via:jarvis from:elias session:...] envelope automatically "
                "using the active assistant session so the recipient can reply "
                "through Jarvis. "
                "Warns if the agent is currently busy."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the target tmux session",
                    },
                    "message": {
                        "type": "string",
                        "description": "Message to send to the agent",
                    },
                },
                "required": ["session_name", "message"],
            },
            category=ToolCategory.BACKEND,
        ),
        send_agent_message,
    )

    logger.info("Registered agent management tools", count=6)
