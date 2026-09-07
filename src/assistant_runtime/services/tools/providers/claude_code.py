"""The Claude Code state-file provider: approvals of plans that agents wrote to their state files."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from assistant_runtime.services.tools.providers._local_sessions import (
    _run_command,
    _validate_session_name,
)

STATE_DIR = Path.home() / ".claude" / "state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_all_state_files(state_dir: Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    """Read all state files and return list of (session_name, state_data)."""
    results = []
    state_dir = state_dir or STATE_DIR
    if not state_dir.is_dir():
        return results

    for state_file in state_dir.glob("*.json"):
        try:
            data = json.loads(state_file.read_text())
            if isinstance(data, dict):
                results.append((state_file.stem, data))
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to read state file", path=str(state_file))
            continue

    return results


def _get_agent_state(session_name: str, state_dir: Path | None = None) -> dict[str, Any] | None:
    """Read a specific agent's state file."""
    state_path = (state_dir or STATE_DIR) / f"{session_name}.json"
    try:
        data = json.loads(state_path.read_text())
        if isinstance(data, dict):
            return data
        return None
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read agent state file", session=session_name)
        return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def list_agent_plans(state_dir: Path | None = None) -> dict[str, Any]:
    """List all agents with plans awaiting approval."""
    all_states = _read_all_state_files(state_dir)

    plans = []
    for session_name, state in all_states:
        if state.get("state") != "plan_waiting":
            continue

        ts = state.get("ts")
        waiting_since = None
        if ts:
            elapsed = int(time.time() - ts)
            minutes = elapsed // 60
            if minutes < 60:
                waiting_since = f"{minutes}m ago"
            else:
                hours = minutes // 60
                waiting_since = f"{hours}h {minutes % 60}m ago"

        plans.append(
            {
                "session_name": session_name,
                "plan_title": state.get("plan_title"),
                "plan_file": state.get("plan_file"),
                "waiting_since": waiting_since,
            }
        )

    return {"plans": plans, "count": len(plans), "success": True}


async def approve_plan(session_name: str, state_dir: Path | None = None) -> dict[str, Any]:
    """Approve a pending plan by sending Shift+Tab to the agent's session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    state = _get_agent_state(session_name, state_dir)
    if not state or state.get("state") != "plan_waiting":
        return {
            "error": f"Agent '{session_name}' is not in plan_waiting state",
            "success": False,
        }

    # Verify tmux session exists
    rc, _, _ = await _run_command(["tmux", "has-session", "-t", session_name])
    if rc != 0:
        return {"error": f"Session '{session_name}' does not exist", "success": False}

    # Send Shift+Tab to approve the plan
    rc, _, stderr = await _run_command(["tmux", "send-keys", "-t", session_name, "\\e[Z"])
    if rc != 0:
        return {"error": f"Failed to send approval: {stderr}", "success": False}

    logger.info("Approved plan", session=session_name)
    return {
        "session_name": session_name,
        "approved": True,
        "success": True,
    }


async def reject_plan(
    session_name: str, reason: str, state_dir: Path | None = None
) -> dict[str, Any]:
    """Reject a pending plan by sending the rejection reason to the agent's session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    if not reason or not reason.strip():
        return {"error": "Rejection reason cannot be empty", "success": False}

    state = _get_agent_state(session_name, state_dir)
    if not state or state.get("state") != "plan_waiting":
        return {
            "error": f"Agent '{session_name}' is not in plan_waiting state",
            "success": False,
        }

    # Verify tmux session exists
    rc, _, _ = await _run_command(["tmux", "has-session", "-t", session_name])
    if rc != 0:
        return {"error": f"Session '{session_name}' does not exist", "success": False}

    # Send rejection reason as text
    rc, _, stderr = await _run_command(
        ["tmux", "send-keys", "-t", session_name, "-l", reason.strip()]
    )
    if rc != 0:
        return {"error": f"Failed to send rejection: {stderr}", "success": False}

    # Press Enter to submit
    rc, _, stderr = await _run_command(["tmux", "send-keys", "-t", session_name, "Enter"])
    if rc != 0:
        return {"error": f"Failed to send Enter: {stderr}", "success": False}

    logger.info("Rejected plan", session=session_name, reason=reason.strip())
    return {
        "session_name": session_name,
        "rejected": True,
        "reason": reason.strip(),
        "success": True,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class StateFileApprovals:
    """The approvals capability over a directory of agent state files."""

    def __init__(self, state_dir: Path | None = None) -> None:
        self._state_dir = state_dir

    async def list_agent_plans(self) -> dict[str, Any]:
        return await list_agent_plans(state_dir=self._state_dir)

    async def approve_plan(self, session_name: str) -> dict[str, Any]:
        return await approve_plan(session_name, state_dir=self._state_dir)

    async def reject_plan(self, session_name: str, reason: str) -> dict[str, Any]:
        return await reject_plan(session_name, reason, state_dir=self._state_dir)
