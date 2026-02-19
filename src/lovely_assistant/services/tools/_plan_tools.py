"""Plan management tools -- list, approve, and reject pending agent plans."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from lovely_assistant.services.tools._agent_tools import _run_command, _validate_session_name
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

STATE_DIR = Path.home() / ".claude" / "state"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_all_state_files() -> list[tuple[str, dict[str, Any]]]:
    """Read all state files and return list of (session_name, state_data)."""
    results = []
    if not STATE_DIR.is_dir():
        return results

    for state_file in STATE_DIR.glob("*.json"):
        try:
            data = json.loads(state_file.read_text())
            if isinstance(data, dict):
                results.append((state_file.stem, data))
        except (json.JSONDecodeError, OSError):
            continue

    return results


def _get_agent_state(session_name: str) -> dict[str, Any] | None:
    """Read a specific agent's state file."""
    state_path = STATE_DIR / f"{session_name}.json"
    try:
        data = json.loads(state_path.read_text())
        if isinstance(data, dict):
            return data
        return None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def list_agent_plans() -> dict[str, Any]:
    """List all agents with plans awaiting approval."""
    all_states = _read_all_state_files()

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


async def approve_plan(session_name: str) -> dict[str, Any]:
    """Approve a pending plan by sending Shift+Tab to the agent's session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    state = _get_agent_state(session_name)
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


async def reject_plan(session_name: str, reason: str) -> dict[str, Any]:
    """Reject a pending plan by sending the rejection reason to the agent's session."""
    error = _validate_session_name(session_name)
    if error:
        return {"error": error, "success": False}

    if not reason or not reason.strip():
        return {"error": "Rejection reason cannot be empty", "success": False}

    state = _get_agent_state(session_name)
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


def register_plan_tools(registry: ToolRegistry) -> None:
    """Register all plan management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="list_agent_plans",
            description=(
                "List all agents that have plans awaiting approval. "
                "Shows session name, plan title, plan file path, and how long "
                "the plan has been waiting."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        list_agent_plans,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="approve_plan",
            description=(
                "Approve a pending plan for an agent. Sends the Shift+Tab key "
                "sequence to the agent's tmux session to trigger plan approval."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session with a pending plan",
                    },
                },
                "required": ["session_name"],
            },
            category=ToolCategory.BACKEND,
        ),
        approve_plan,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="reject_plan",
            description=(
                "Reject a pending plan for an agent. Sends the rejection reason "
                "as text to the agent's tmux session."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "session_name": {
                        "type": "string",
                        "description": "Name of the tmux session with a pending plan",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for rejecting the plan",
                    },
                },
                "required": ["session_name", "reason"],
            },
            category=ToolCategory.BACKEND,
        ),
        reject_plan,
    )

    logger.info("Registered plan management tools", count=3)
