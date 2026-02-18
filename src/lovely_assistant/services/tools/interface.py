"""ToolService -- tool system facade with lifecycle management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolError
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


class ToolService:
    """Tool system facade. Implements LifecycleAware."""

    def __init__(self, config: ToolConfig) -> None:
        self._config = config
        self._registry: ToolRegistry | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize registry and register placeholder tools."""
        self._registry = ToolRegistry(self._config)
        self._register_placeholder_tools()
        self._started = True
        logger.info(
            "Tool service started",
            backend_tools=len(self._registry._backend_definitions),
            frontend_tools=len(self._registry._frontend_definitions),
        )

    async def stop(self) -> None:
        """Shutdown the tool service."""
        self._registry = None
        self._started = False
        logger.info("Tool service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        if not self._started or self._registry is None:
            return {"healthy": False}
        return {
            "healthy": True,
            "backend_tools": len(self._registry._backend_definitions),
            "frontend_tools": len(self._registry._frontend_definitions),
        }

    def build_toolset(self, machine_state: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request."""
        self._ensure_started()
        return self._registry.build_toolset(machine_state)

    def get_available_tools(self, machine_state: dict[str, Any] | None = None) -> ToolSet:
        """List tools available for a given machine state."""
        self._ensure_started()
        return self._registry.get_available_tools(machine_state)

    def validate_tool_call(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if a tool name is registered."""
        self._ensure_started()
        return self._registry.validate_tool_call(tool_name, args)

    def register_backend_tool(self, definition: ToolDefinition, handler: Callable) -> None:
        """Register a backend tool with its handler."""
        self._ensure_started()
        self._registry.register_backend_tool(definition, handler)

    def register_frontend_tool(self, definition: ToolDefinition) -> None:
        """Register a frontend tool definition."""
        self._ensure_started()
        self._registry.register_frontend_tool(definition)

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._registry is None:
            raise ToolError("Tool service not started")

    def _register_placeholder_tools(self) -> None:
        """Register minimal placeholder tools for testing both code paths."""

        # Backend placeholder: get_time
        async def get_time() -> str:
            """Get the current UTC time in ISO format."""
            return datetime.now(UTC).isoformat()

        self._registry.register_backend_tool(
            ToolDefinition(
                name="get_time",
                description="Get the current UTC time in ISO format.",
                parameters_schema={"type": "object", "properties": {}},
                category=ToolCategory.BACKEND,
            ),
            get_time,
        )

        # Frontend placeholder: ui_notify
        self._registry.register_frontend_tool(
            ToolDefinition(
                name="ui_notify",
                description="Send a notification to the user interface.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Notification message",
                        },
                        "level": {
                            "type": "string",
                            "enum": ["info", "warning", "error"],
                            "description": "Notification severity level",
                        },
                    },
                    "required": ["message"],
                },
                category=ToolCategory.FRONTEND,
            ),
        )
