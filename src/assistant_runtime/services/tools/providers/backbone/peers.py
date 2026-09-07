"""Agent management tools -- list, inspect, start, and message AI agents."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger

from assistant_runtime.services.tools._request_context import get_current_assistant_session_id
from assistant_runtime.services.tools.providers._local_sessions import (
    MAX_SESSION_NAME_LENGTH,
    SESSION_NAME_PATTERN,
    _run_command,
    _validate_session_name,
)
from assistant_runtime.services.tools.providers.backbone._client import (
    backbone_error,
    backbone_request,
)
from assistant_runtime.services.tools.providers.backbone._registry_cache import get_registry_cache

__all__ = ["MAX_SESSION_NAME_LENGTH", "SESSION_NAME_PATTERN", "BackbonePeers"]

STATE_DIR = Path.home() / ".claude" / "state"  # default when the provider is built without one


OPERATOR_NAME_ENV = "ASSISTANT_OPERATOR_NAME"


def _operator_name() -> str:
    """Name used as the human sender in reply envelopes (configurable, defaults to 'operator')."""
    return os.environ.get(OPERATOR_NAME_ENV, "operator")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_state_file(session_name: str, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Read an agent state file. Returns None if missing or invalid."""
    state_path = (state_dir or STATE_DIR) / f"{session_name}.json"
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


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def list_agents(state_dir: Path | None = None) -> dict[str, Any]:
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
        state = _read_state_file(name, state_dir)
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


async def get_active_agents(
    infrastructure_sessions: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """List active AI agents, excluding infrastructure and offline sessions.

    Uses the backbone registry API which provides runtime, state, and
    entity metadata. Filters out the configured infrastructure sessions,
    unknown state, and offline agents.
    """
    cache = get_registry_cache()
    agents = await cache.get_agents()
    if agents is None:
        return {"error": "Backbone agent registry unavailable", "success": False}

    active = []
    for agent in agents:
        session = agent.get("session", "")
        # Skip infrastructure
        if session in infrastructure_sessions:
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


async def check_agent_state(session_name: str, state_dir: Path | None = None) -> dict[str, Any]:
    """Check detailed state of a specific agent session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    state = _read_state_file(session_name, state_dir)

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


async def stop_agent(session_name: str, state_dir: Path | None = None) -> dict[str, Any]:
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
    state = _read_state_file(session_name, state_dir)
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


async def send_agent_message(
    session_name: str, message: str, state_dir: Path | None = None
) -> dict[str, Any]:
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
    state = _read_state_file(session_name, state_dir)
    state_warning = None
    if state and state.get("state") in ("processing", "busy"):
        state_warning = f"Agent is currently {state['state']} — message will be queued"

    reply_session_id = get_current_assistant_session_id()
    if not reply_session_id:
        return {
            "error": "Cannot send assistant-routed message without assistant session context",
            "success": False,
        }

    # Send with reply-safe assistant envelope
    envelope = (
        f"[via:assistant from:{_operator_name()} session:{reply_session_id}] {message.strip()}"
    )
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


class BackbonePeers:
    """The peers capability served by agent-backbone plus the local session state."""

    def __init__(
        self,
        *,
        infrastructure_sessions: frozenset[str] = frozenset(),
        state_dir: Path | None = None,
    ) -> None:
        self._infrastructure = infrastructure_sessions
        self._state_dir = state_dir

    async def list_agents(self) -> dict[str, Any]:
        return await list_agents(state_dir=self._state_dir)

    async def get_active_agents(self) -> dict[str, Any]:
        return await get_active_agents(infrastructure_sessions=self._infrastructure)

    async def check_agent_state(self, session_name: str) -> dict[str, Any]:
        return await check_agent_state(session_name, state_dir=self._state_dir)

    async def start_agent(
        self,
        session_name: str,
        runtime: str = "claude",
        model: str | None = None,
        resume: bool = False,
        initial_prompt: str = "",
    ) -> dict[str, Any]:
        return await start_agent(
            session_name, runtime=runtime, model=model, resume=resume, initial_prompt=initial_prompt
        )

    async def stop_agent(self, session_name: str) -> dict[str, Any]:
        return await stop_agent(session_name, state_dir=self._state_dir)

    async def send_agent_message(self, session_name: str, message: str) -> dict[str, Any]:
        return await send_agent_message(session_name, message, state_dir=self._state_dir)
