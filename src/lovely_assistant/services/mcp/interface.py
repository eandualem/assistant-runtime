"""MCPService — MCP server lifecycle and toolset provider. Implements LifecycleAware."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic_ai.mcp import load_mcp_servers


class MCPService:
    """MCP server lifecycle and toolset provider. Implements LifecycleAware.

    Loads MCP server definitions from a JSON config file (Pydantic AI native format),
    enters each server's async context manager on start(), and provides the live
    servers as toolsets for Agent() construction.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path
        self._servers: list[Any] = []  # All configured MCPServer instances
        self._live_servers: list[Any] = []  # Successfully started servers
        self._failed: list[dict[str, str]] = []  # Name + reason for failures
        self._exit_stack: AsyncExitStack | None = None
        self._detailed_cache: list[dict[str, Any]] | None = None
        self._started = False

    async def start(self) -> None:
        """Load and connect to configured MCP servers.

        If no config file is set, the service starts with no servers (valid state).
        Individual server failures are logged but don't prevent other servers
        from starting.
        """
        if self._config_path is None:
            logger.info("MCP service started (no config file)")
            self._started = True
            return

        try:
            self._servers = load_mcp_servers(self._config_path)
        except Exception as e:
            logger.error(
                "Failed to load MCP server config", path=str(self._config_path), error=str(e)
            )
            self._started = True
            return

        if not self._servers:
            logger.info("MCP service started (no servers configured)")
            self._started = True
            return

        # Enter each server's context manager individually
        self._exit_stack = AsyncExitStack()
        for server in self._servers:
            name = getattr(server, "id", None) or getattr(server, "tool_prefix", None) or "unknown"
            try:
                await self._exit_stack.enter_async_context(server)
                self._live_servers.append(server)
                logger.info("MCP server connected", name=name)
            except Exception as e:
                self._failed.append({"name": name, "reason": str(e)})
                logger.warning("MCP server failed to start", name=name, error=str(e))

        self._started = True
        logger.info(
            "MCP service started",
            connected=len(self._live_servers),
            failed=len(self._failed),
        )

    async def stop(self) -> None:
        """Close all server connections."""
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("Error closing MCP exit stack", error=str(e))
            self._exit_stack = None
        self._live_servers.clear()
        self._failed.clear()
        self._servers.clear()
        self._detailed_cache = None
        self._started = False
        logger.info("MCP service stopped")

    async def health_check(self) -> dict[str, Any]:
        """Report connection status."""
        if not self._started:
            return {"healthy": False}
        return {
            "healthy": True,
            "connected": len(self._live_servers),
            "failed": self._failed,
            "server_names": [
                getattr(s, "id", None) or getattr(s, "tool_prefix", None) or "unknown"
                for s in self._live_servers
            ],
        }

    def get_toolsets(self) -> list[Any]:
        """Return live MCP servers as Pydantic AI toolsets.

        MCPServer instances ARE AbstractToolset subclasses — they can be passed
        directly to Agent(toolsets=...).
        """
        return list(self._live_servers)

    def get_server_summary(self) -> list[dict[str, Any]]:
        """Compact summary of connected servers for prompt builder.

        Returns a list of dicts with name and tool_prefix per live server.
        Prefer get_detailed_summary() for richer tool-aware data.
        """
        summaries: list[dict[str, Any]] = []
        for server in self._live_servers:
            name = getattr(server, "id", None) or getattr(server, "tool_prefix", None) or "unknown"
            summaries.append({"name": name})
        return summaries

    async def get_detailed_summary(self) -> list[dict[str, Any]]:
        """Query live servers for tool lists. Cached after first call.

        Returns a list of dicts with name, tools (list of tool names), and
        tool_count per live server.
        """
        if self._detailed_cache is not None:
            return self._detailed_cache

        summaries: list[dict[str, Any]] = []
        for server in self._live_servers:
            name = getattr(server, "id", None) or getattr(server, "tool_prefix", None) or "unknown"
            tools: list[str] = []
            try:
                tool_defs = await server.list_tools()
                tools = [t.name for t in tool_defs]
            except Exception as e:
                logger.warning("Failed to list tools for MCP server", name=name, error=str(e))
            summaries.append({"name": name, "tools": tools, "tool_count": len(tools)})

        self._detailed_cache = summaries
        return summaries
