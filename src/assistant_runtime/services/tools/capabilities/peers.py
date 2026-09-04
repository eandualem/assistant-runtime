"""Peers: the other agents the assistant works with.

List them, inspect one, start or stop one, send one a message.

A provider implements ``PeersProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class PeersProvider(Protocol):
    """What a provider of the peers capability implements."""

    async def list_agents(self) -> dict[str, Any]: ...

    async def get_active_agents(self) -> dict[str, Any]: ...

    async def check_agent_state(self, session_name: str) -> dict[str, Any]: ...

    async def start_agent(
        self,
        session_name: str,
        runtime: str = "claude",
        model: str | None = None,
        resume: bool = False,
        initial_prompt: str = "",
    ) -> dict[str, Any]: ...

    async def stop_agent(self, session_name: str) -> dict[str, Any]: ...

    async def send_agent_message(self, session_name: str, message: str) -> dict[str, Any]: ...


def register_peers_tools(registry: ToolRegistry, provider: PeersProvider) -> None:
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
        provider.list_agents,
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
        provider.get_active_agents,
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
        provider.check_agent_state,
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
        provider.start_agent,
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
        provider.stop_agent,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="send_agent_message",
            description=(
                "Send a message to a running agent session. Prepends the "
                "[via:assistant from:<operator> session:...] envelope automatically "
                "using the active assistant session so the recipient can reply "
                "through the assistant. "
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
        provider.send_agent_message,
    )

    logger.info("Registered agent management tools", count=6)
