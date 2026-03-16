"""Internal tool registry -- manages tool definitions and builds Pydantic AI toolsets."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic_ai.toolsets import FunctionToolset

from lovely_assistant.base.resilience import retry_with_backoff
from lovely_assistant.services.tools._frontend_tools import (
    build_frontend_toolset,
    get_frontend_definitions,
)
from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolValidationError
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

_CORE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_time",
        "manage_notes",
        "manage_artifacts",
        "add_schedule_item",
        "remove_schedule_item",
        "toggle_schedule_item",
        "get_delivery_status",
        "get_recent_deliveries",
        "get_failed_deliveries",
        "get_agent_activity",
        "get_activity_timeline",
        "create_swarm",
        "list_swarms",
        "get_swarm_detail",
        "update_worker_status",
        "broadcast_to_swarm",
        "complete_swarm",
        "generate_image",
        "generate_video",
        "run_subagent",
        "respond_telegram",
        "look_at_screen",
        # Skill tools are cross-cutting — inspect skills from any page
        "list_skills",
        "read_skill",
        # Agent tools are cross-cutting — check/manage agents from any page
        "list_agents",
        "get_active_agents",
        "check_agent_state",
        "start_agent",
        "stop_agent",
        "send_agent_message",
    }
)
_GITHUB_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "create_issue",
        "search_issues",
        "get_issue_details",
        "comment_on_issue",
        "close_issue",
    }
)
_MEETING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "create_meeting_room",
        "list_meeting_rooms",
        "send_meeting_message",
        "update_meeting_state",
    }
)
_PLAN_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_agent_plans",
        "approve_plan",
        "reject_plan",
    }
)
_REPO_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "onboard_repo",
        "check_repo_status",
        "list_repos",
    }
)
_AGENT_PAGES: frozenset[str] = frozenset({"agents", "sessions"})
_GITHUB_PAGES: frozenset[str] = frozenset({"tasks"})
_MEETING_PAGES: frozenset[str] = frozenset({"meetings"})
_REPO_PAGES: frozenset[str] = frozenset({"repos"})
_CORE_ONLY_PAGES: frozenset[str] = frozenset({"flows"})

# Tools that invalidate frontend data domains when executed.
# The dashboard uses this to know which cached data to refresh.
TOOL_INVALIDATES: dict[str, list[str]] = {
    "create_meeting_room": ["meetings"],
    "update_meeting_state": ["meetings"],
    "send_meeting_message": ["meetings"],
    "create_issue": ["tasks"],
    "comment_on_issue": ["tasks"],
    "close_issue": ["tasks"],
    "start_agent": ["agents"],
    "stop_agent": ["agents"],
    "send_agent_message": ["agents"],
    "approve_plan": ["agents"],
    "reject_plan": ["agents"],
    "manage_notes": ["notes"],
    "manage_artifacts": ["artifacts"],
    "add_schedule_item": ["schedule"],
    "remove_schedule_item": ["schedule"],
    "toggle_schedule_item": ["schedule"],
    "onboard_repo": ["repos"],
}


def get_tool_invalidates(tool_name: str) -> list[str] | None:
    """Return the list of data domains invalidated by a tool, or None."""
    result = TOOL_INVALIDATES.get(tool_name)
    return result if result else None


class ToolRegistry:
    """Manages tool definitions and handlers, builds Pydantic AI toolsets per request."""

    def __init__(self, config: ToolConfig) -> None:
        self._config = config
        self._backend_handlers: dict[str, Callable] = {}
        self._backend_definitions: dict[str, ToolDefinition] = {}
        self._frontend_definitions: list[ToolDefinition] = []

    def register_backend_tool(self, definition: ToolDefinition, handler: Callable) -> None:
        """Register a backend tool with its async handler function."""
        if definition.category != ToolCategory.BACKEND:
            raise ToolValidationError(
                f"Expected backend tool, got category '{definition.category}'"
            )
        if definition.name in self._backend_definitions:
            raise ToolValidationError(f"Backend tool '{definition.name}' already registered")
        self._backend_definitions[definition.name] = definition
        self._backend_handlers[definition.name] = handler
        logger.debug("Registered backend tool", tool=definition.name)

    def register_frontend_tools(self) -> None:
        """Register frontend tools from the built-in schema definitions."""
        self._frontend_definitions = get_frontend_definitions()
        logger.debug(
            "Registered frontend tools",
            count=len(self._frontend_definitions),
            names=[d.name for d in self._frontend_definitions],
        )

    @staticmethod
    def _wrap_handler(handler: Callable, tool_name: str) -> Callable:
        """Wrap a tool handler with retry on transient errors and a safety net.

        Retries once (2 total attempts) on ConnectionError/TimeoutError before
        falling through to the structured error response.
        """

        @retry_with_backoff(
            max_attempts=2,
            min_wait=0.5,
            max_wait=5.0,
            retry_on=(ConnectionError, TimeoutError),
            name=f"tool:{tool_name}",
        )
        async def _retryable(*args: Any, **kwargs: Any) -> Any:
            return await handler(*args, **kwargs)

        @functools.wraps(handler)
        async def _safe_handler(*args: Any, **kwargs: Any) -> Any:
            try:
                return await _retryable(*args, **kwargs)
            except Exception as e:
                logger.error(
                    "[TOOLS] Unhandled tool exception",
                    tool=tool_name,
                    error_type=e.__class__.__name__,
                    error=str(e),
                )
                return {
                    "success": False,
                    "error": f"Internal error: {e}",
                    "error_code": "TOOL_EXECUTION_ERROR",
                }

        return _safe_handler

    def build_toolset(self, machine_state: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request.

        Returns list of AbstractToolset instances:
        - FunctionToolset for backend tools (with real handlers, wrapped with safety net)
        - ExternalToolset for frontend tools (deferred execution via DeferredToolRequests)
        """
        available = self._resolve_available_tools(machine_state)
        toolsets: list = []

        if available.backend_tools:
            func_toolset = FunctionToolset()
            for defn in available.backend_tools:
                handler = self._backend_handlers[defn.name]
                safe_handler = self._wrap_handler(handler, defn.name)
                func_toolset.add_function(
                    safe_handler,
                    name=defn.name,
                    description=defn.description,
                )
            toolsets.append(func_toolset)

        # Frontend tools — always appended, bypass page filtering
        if self._frontend_definitions:
            ext_toolset = build_frontend_toolset()
            if ext_toolset is not None:
                toolsets.append(ext_toolset)

        logger.debug(
            "[TOOLS] Built toolsets",
            backend=len(available.backend_tools),
            frontend=len(self._frontend_definitions),
            toolsets=len(toolsets),
        )
        return toolsets

    def build_subagent_toolset(self) -> list:
        """Build toolsets for subagent execution — backend tools only, excluding run_subagent.

        Returns a list with a single FunctionToolset containing all backend tools
        except run_subagent (prevents recursion). No frontend tools — subagents
        don't interact with the UI.
        """
        toolsets: list = []
        backend_defs = [
            defn for defn in self._backend_definitions.values() if defn.name != "run_subagent"
        ]

        if backend_defs:
            func_toolset = FunctionToolset()
            for defn in backend_defs:
                handler = self._backend_handlers[defn.name]
                safe_handler = self._wrap_handler(handler, defn.name)
                func_toolset.add_function(
                    safe_handler,
                    name=defn.name,
                    description=defn.description,
                )
            toolsets.append(func_toolset)

        logger.debug(
            "[TOOLS] Built subagent toolsets",
            backend=len(backend_defs),
            excluded="run_subagent",
        )
        return toolsets

    def get_available_tools(self, machine_state: dict[str, Any] | None = None) -> ToolSet:
        """List tools available for a given machine state."""
        return self._resolve_available_tools(machine_state)

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if a tool name is registered."""
        return tool_name in self._backend_definitions

    def get_tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._backend_definitions.keys())

    def backend_tool_count(self) -> int:
        """Number of registered backend tools."""
        return len(self._backend_definitions)

    def frontend_tool_count(self) -> int:
        """Number of registered frontend tools."""
        return len(self._frontend_definitions)

    def configure_handler_deps(self, handler_name: str, deps: dict) -> None:
        """Set runtime dependency dict on a registered backend handler."""
        handler = self._backend_handlers.get(handler_name)
        if handler is not None:
            handler._handler_deps = deps

    def _resolve_available_tools(self, machine_state: dict[str, Any] | None = None) -> ToolSet:
        """Determine which tools are available based on the active page.

        Filtering logic:
        - No machine_state or no active_page → all tools
        - home page or unknown page → all tools
        - agents/sessions pages → core + plan tools
        - tasks page → core + github tools
        - meetings page → core + meeting tools
        - repos page → core + repo tools
        - flows page → core only

        Agent tools (list_agents, get_active_agents, check_agent_state,
        start/stop/send) are in core — available on every page.
        """
        backend = list(self._backend_definitions.values())
        total_before = len(backend)
        page_name: str | None = None

        # Determine if we should filter
        if machine_state and isinstance(machine_state.get("active_page"), dict):
            page_name = machine_state["active_page"].get("name")
            if page_name and page_name != "home":
                if page_name in _AGENT_PAGES:
                    allowed = _CORE_TOOL_NAMES | _PLAN_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                elif page_name in _GITHUB_PAGES:
                    allowed = _CORE_TOOL_NAMES | _GITHUB_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                elif page_name in _MEETING_PAGES:
                    allowed = _CORE_TOOL_NAMES | _MEETING_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                elif page_name in _REPO_PAGES:
                    allowed = _CORE_TOOL_NAMES | _REPO_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                elif page_name in _CORE_ONLY_PAGES:
                    backend = [t for t in backend if t.name in _CORE_TOOL_NAMES]
                # else: unknown page → all tools (no filtering)

        total = len(backend)
        if total > self._config.max_tools_per_request:
            logger.warning(
                "Tool count exceeds max_tools_per_request",
                total=total,
                max=self._config.max_tools_per_request,
            )

        return ToolSet(
            backend_tools=backend,
            frontend_tools=self._frontend_definitions,
            page=page_name,
            filtered_out_count=total_before - total,
        )
