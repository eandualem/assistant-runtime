"""ToolService -- tool system facade with lifecycle management."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from lovely_assistant.services.tools._agent_tools import register_agent_tools
from lovely_assistant.services.tools._artifact_tools import register_artifact_tools
from lovely_assistant.services.tools._github_tools import register_github_tools
from lovely_assistant.services.tools._media_tools import register_media_tools
from lovely_assistant.services.tools._meeting_tools import register_meeting_tools
from lovely_assistant.services.tools._notes_tools import register_notes_tools
from lovely_assistant.services.tools._plan_tools import register_plan_tools
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._repo_tools import register_repo_tools
from lovely_assistant.services.tools._schedule_tools import register_schedule_tools
from lovely_assistant.services.tools._screen_tools import register_screen_tools
from lovely_assistant.services.tools._skill_tools import register_skill_tools
from lovely_assistant.services.tools._subagent_tools import register_subagent_tools
from lovely_assistant.services.tools._telegram_tools import register_telegram_tools
from lovely_assistant.services.tools._telemetry_tools import register_telemetry_tools
from lovely_assistant.services.tools._video_tools import register_video_tools
from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolError
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


class ToolService:
    """Tool system facade. Implements LifecycleAware."""

    def __init__(
        self,
        config: ToolConfig,
        media_service: Any | None = None,
        llm_service: Any | None = None,
        mcp_service: Any | None = None,
        database_service: Any | None = None,
    ) -> None:
        self._config = config
        self._media_service = media_service
        self._llm_service = llm_service
        self._mcp_service = mcp_service
        self._database_service = database_service
        self._runtime_settings: object | None = None
        self._registry: ToolRegistry | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize registry and register default tools."""
        self._registry = ToolRegistry(self._config)
        self._register_default_tools()
        self._started = True
        logger.info(
            "Tool service started",
            backend_tools=self._registry.backend_tool_count(),
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
            "backend_tools": self._registry.backend_tool_count(),
            "frontend_tools": self._registry.frontend_tool_count(),
        }

    def build_toolset(self, machine_state: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request.

        Includes MCP server toolsets (always available, not page-filtered).
        """
        self._ensure_started()
        toolsets = self._registry.build_toolset(machine_state)
        if self._mcp_service is not None:
            toolsets.extend(self._mcp_service.get_toolsets())
        return toolsets

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

    def get_subagent_toolsets(self) -> list:
        """Build toolsets for subagent execution (backend only, no run_subagent)."""
        self._ensure_started()
        return self._registry.build_subagent_toolset()

    def set_runtime_settings(self, runtime_settings: object | None) -> None:
        """Attach live runtime settings and propagate to subagent handler deps."""
        self._runtime_settings = runtime_settings
        self._configure_subagent_handler_deps()

    async def get_mcp_summary(self) -> list[dict[str, Any]] | None:
        """Return MCP server summary for prompt builder, or None if no MCP service."""
        if self._mcp_service is None:
            return None
        try:
            summary = await self._mcp_service.get_detailed_summary()
        except Exception as exc:
            logger.warning("MCP detailed summary failed, falling back", error=str(exc))
            summary = self._mcp_service.get_server_summary()
        return summary or None

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._registry is None:
            raise ToolError("Tool service not started")

    def _register_default_tools(self) -> None:
        """Register built-in tools: placeholders and agent management tools."""

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

        # Frontend tools (always available, bypass page filtering)
        self._registry.register_frontend_tools()

        # Agent management tools
        register_agent_tools(self._registry)

        # Notes management tool
        register_notes_tools(self._registry)

        # GitHub issue management tools
        register_github_tools(self._registry)

        # Meeting room management tools
        register_meeting_tools(self._registry)

        # Schedule management tools
        register_schedule_tools(self._registry)

        # Backbone telemetry and delivery status tools
        register_telemetry_tools(self._registry)

        # Telegram messaging tools
        register_telegram_tools(self._registry)

        # Plan management tools
        register_plan_tools(self._registry)

        # Repo management tools
        register_repo_tools(self._registry)

        # Screen inspection tools (look_at_screen)
        register_screen_tools(self._registry)

        # Skill management tools (read-only filesystem access)
        register_skill_tools(self._registry)

        # Artifact management tools
        register_artifact_tools(self._registry)
        self._configure_artifact_handler_deps()

        # Subagent tools (only if llm_service is available)
        if self._llm_service is not None:
            register_subagent_tools(self._registry)
            self._configure_subagent_handler_deps()

        # Media generation tools (only if media service is available)
        if self._media_service is not None:
            register_media_tools(self._registry, self._media_service)
            register_video_tools(self._registry, self._media_service)

    def _configure_artifact_handler_deps(self) -> None:
        """Configure runtime dependencies consumed by manage_artifacts handler."""
        if self._registry is None:
            return
        self._registry.configure_handler_deps(
            "manage_artifacts",
            {"database_service": self._database_service},
        )

    def _configure_subagent_handler_deps(self) -> None:
        """Configure runtime dependencies consumed by run_subagent handler."""
        if self._registry is None:
            return
        self._registry.configure_handler_deps(
            "run_subagent",
            {
                "llm_service": self._llm_service,
                "get_backend_toolsets": self._registry.build_subagent_toolset,
                "runtime_settings": getattr(self, "_runtime_settings", None),
            },
        )
