"""Factory for MCP service lifecycle registration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.services.mcp.interface import MCPService

# Default config file: mcp_servers.json in the working directory (gitignored; copy
# from mcp_servers.example.json). Resolved at startup, not at import, so it works
# the same from a source checkout and from an installed package. No servers are
# started when the file does not exist.
DEFAULT_CONFIG_NAME = "mcp_servers.json"


def default_config_path() -> Path:
    """Where the MCP config is looked up when MCP_CONFIG_PATH is not set."""
    return Path.cwd() / DEFAULT_CONFIG_NAME


async def register_mcp(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create MCPService, store on app_state, register with lifecycle."""
    config_path_str = os.environ.get("MCP_CONFIG_PATH")
    config_path = Path(config_path_str) if config_path_str else default_config_path()

    service = MCPService(config_path=config_path if config_path.exists() else None)
    app_state.mcp_service = service
    await lifecycle.register("mcp_service", service)
    logger.info("MCP module registered", config_path=str(config_path))
