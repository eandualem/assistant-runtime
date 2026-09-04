"""ToolService -- tool system facade with lifecycle management."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.builtin import register_builtin_tools
from assistant_runtime.services.tools.capabilities import register_capabilities
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolError
from assistant_runtime.services.tools.models import ToolDefinition, ToolSet


class ToolService:
    """Tool system facade. Implements LifecycleAware."""

    def __init__(
        self,
        config: ToolConfig,
        media_service: Any | None = None,
        llm_service: Any | None = None,
        mcp_service: Any | None = None,
        database_service: Any | None = None,
        providers: dict[str, Any] | None = None,
    ) -> None:
        self._config = config
        self._providers = providers or {}
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
        capabilities = self._register_default_tools()
        self._started = True
        logger.info(
            "Tool service started",
            backend_tools=self._registry.backend_tool_count(),
            capabilities=capabilities,
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
            "host_tools": self._registry.host_tool_count(),
        }

    def build_toolset(self, host_context: dict[str, Any] | None = None) -> list:
        """Build Pydantic AI toolsets for a request.

        Includes MCP server toolsets (always available, not page-scoped).
        """
        self._ensure_started()
        toolsets = self._registry.build_toolset(host_context)
        if self._mcp_service is not None:
            toolsets.extend(self._mcp_service.get_toolsets())
        return toolsets

    def get_available_tools(self, host_context: dict[str, Any] | None = None) -> ToolSet:
        """List tools available for a given host context."""
        self._ensure_started()
        return self._registry.get_available_tools(host_context)

    def warm_host_context(self, host_context: dict[str, Any] | None = None) -> None:
        """Precompute page-scoped tool availability and toolsets."""
        self._ensure_started()
        self._registry.warm_host_context(host_context)

    def is_host_tool(self, tool_name: str) -> bool:
        """Whether the host application executes this tool (deferred call)."""
        self._ensure_started()
        return self._registry.is_host_tool(tool_name)

    def get_tool_invalidates(self, tool_name: str) -> list[str] | None:
        """Host data domains a tool invalidates, from configuration."""
        self._ensure_started()
        return self._registry.invalidates_for(tool_name)

    def register_backend_tool(self, definition: ToolDefinition, handler: Callable) -> None:
        """Register a backend tool with its handler."""
        self._ensure_started()
        self._registry.register_backend_tool(definition, handler)

    def get_subagent_toolsets(self) -> list:
        """Build toolsets for subagent execution (backend only, no run_subagent)."""
        self._ensure_started()
        return self._registry.build_subagent_toolset()

    def set_runtime_settings(self, runtime_settings: object | None) -> None:
        """Attach live runtime settings; the subagent tool reads them on each call."""
        self._runtime_settings = runtime_settings

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

    def _register_default_tools(self) -> list[str]:
        """Built-in tools, host tools, then one capability per configured provider."""
        register_builtin_tools(
            self._registry,
            database_service=self._database_service,
            llm_service=self._llm_service,
            media_service=self._media_service,
            backend_toolsets=self._registry.build_subagent_toolset,
            runtime_settings=lambda: self._runtime_settings,
        )
        # Host tools from configuration (always available, bypass page scoping)
        self._registry.register_host_tools()
        return register_capabilities(self._registry, self._providers)
