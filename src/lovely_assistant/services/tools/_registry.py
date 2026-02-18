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
        """Determine which tools are available for the given machine state.

        Stub: returns all registered tools. Full state-driven gating deferred.
        """
        backend = list(self._backend_definitions.values())
        frontend = list(self._frontend_definitions.values())

        total = len(backend) + len(frontend)
        if total > self._config.max_tools_per_request:
            logger.warning(
                "Tool count exceeds max_tools_per_request",
                total=total,
                max=self._config.max_tools_per_request,
            )

        return ToolSet(backend_tools=backend, frontend_tools=frontend)
