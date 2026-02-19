"""Internal tool registry -- manages tool definitions and builds Pydantic AI toolsets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger
from pydantic_ai.tools import ToolDefinition as PydanticToolDef
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolValidationError
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet

_CORE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "get_time",
        "manage_notes",
        "ui_notify",
        "navigate",
        "add_schedule_item",
        "remove_schedule_item",
        "toggle_schedule_item",
    }
)
_AGENT_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "list_agents",
        "check_agent_state",
        "start_agent",
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
_AGENT_PAGES: frozenset[str] = frozenset({"agents", "sessions"})
_GITHUB_PAGES: frozenset[str] = frozenset({"tasks"})
_MEETING_PAGES: frozenset[str] = frozenset({"meetings"})
_CORE_ONLY_PAGES: frozenset[str] = frozenset({"flows", "repos"})


class ToolRegistry:
    """Manages tool definitions and handlers, builds Pydantic AI toolsets per request."""

    def __init__(self, config: ToolConfig) -> None:
        self._config = config
        self._backend_handlers: dict[str, Callable] = {}
        self._backend_definitions: dict[str, ToolDefinition] = {}
        self._frontend_definitions: dict[str, ToolDefinition] = {}

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

    def register_frontend_tool(self, definition: ToolDefinition) -> None:
        """Register a frontend tool definition (no handler -- deferred to frontend)."""
        if definition.category != ToolCategory.FRONTEND:
            raise ToolValidationError(
                f"Expected frontend tool, got category '{definition.category}'"
            )
        if definition.name in self._frontend_definitions:
            raise ToolValidationError(f"Frontend tool '{definition.name}' already registered")
        self._frontend_definitions[definition.name] = definition
        logger.debug("Registered frontend tool", tool=definition.name)

    def build_toolset(self, machine_state: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request.

        Returns list of AbstractToolset instances:
        - FunctionToolset for backend tools (with real handlers)
        - ExternalToolset for frontend tools (deferred via SSE)
        """
        available = self._resolve_available_tools(machine_state)
        toolsets: list = []

        # Backend tools -> FunctionToolset
        if available.backend_tools:
            func_toolset = FunctionToolset()
            for defn in available.backend_tools:
                handler = self._backend_handlers[defn.name]
                func_toolset.add_function(
                    handler,
                    name=defn.name,
                    description=defn.description,
                )
            toolsets.append(func_toolset)

        # Frontend tools -> ExternalToolset (only if enabled)
        if self._config.enable_frontend_tools and available.frontend_tools:
            pydantic_defs = [
                PydanticToolDef(
                    name=defn.name,
                    parameters_json_schema=defn.parameters_schema,
                    description=defn.description,
                )
                for defn in available.frontend_tools
            ]
            toolsets.append(ExternalToolset(tool_defs=pydantic_defs))

        logger.debug(
            "Built toolsets",
            backend=len(available.backend_tools),
            frontend=len(available.frontend_tools),
            toolsets=len(toolsets),
        )
        return toolsets

    def get_available_tools(self, machine_state: dict[str, Any] | None = None) -> ToolSet:
        """List tools available for a given machine state."""
        return self._resolve_available_tools(machine_state)

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if a tool name is registered."""
        return tool_name in self._backend_definitions or tool_name in self._frontend_definitions

    def get_tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._backend_definitions.keys()) + list(self._frontend_definitions.keys())

    def _resolve_available_tools(self, machine_state: dict[str, Any] | None = None) -> ToolSet:
        """Determine which tools are available based on the active page.

        Filtering logic:
        - No machine_state or no active_page → all tools
        - home page or unknown page → all tools
        - agents/sessions pages → core + agent tools
        - tasks/meetings/flows/repos → core tools only
        """
        backend = list(self._backend_definitions.values())
        frontend = list(self._frontend_definitions.values())
        total_before = len(backend) + len(frontend)
        page_name: str | None = None

        # Determine if we should filter
        if machine_state and isinstance(machine_state.get("active_page"), dict):
            page_name = machine_state["active_page"].get("name")
            if page_name and page_name != "home":
                if page_name in _AGENT_PAGES:
                    allowed = _CORE_TOOL_NAMES | _AGENT_TOOL_NAMES | _PLAN_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                    frontend = [t for t in frontend if t.name in allowed]
                elif page_name in _GITHUB_PAGES:
                    allowed = _CORE_TOOL_NAMES | _GITHUB_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                    frontend = [t for t in frontend if t.name in allowed]
                elif page_name in _MEETING_PAGES:
                    allowed = _CORE_TOOL_NAMES | _MEETING_TOOL_NAMES
                    backend = [t for t in backend if t.name in allowed]
                    frontend = [t for t in frontend if t.name in allowed]
                elif page_name in _CORE_ONLY_PAGES:
                    backend = [t for t in backend if t.name in _CORE_TOOL_NAMES]
                    frontend = [t for t in frontend if t.name in _CORE_TOOL_NAMES]
                # else: unknown page → all tools (no filtering)

        total = len(backend) + len(frontend)
        if total > self._config.max_tools_per_request:
            logger.warning(
                "Tool count exceeds max_tools_per_request",
                total=total,
                max=self._config.max_tools_per_request,
            )

        return ToolSet(
            backend_tools=backend,
            frontend_tools=frontend,
            page=page_name,
            filtered_out_count=total_before - total,
        )
