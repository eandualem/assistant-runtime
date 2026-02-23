"""Factory for MCP service lifecycle registration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.services.mcp.interface import MCPService

# Default config file location: repo root / mcp_servers.json
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[4] / "mcp_servers.json"


async def register_mcp(app_state: Any, lifecycle: LifecycleManager) -> None:
    """Create MCPService, store on app_state, register with lifecycle."""
    config_path_str = os.environ.get("MCP_CONFIG_PATH")
    config_path = Path(config_path_str) if config_path_str else _DEFAULT_CONFIG_PATH

    service = MCPService(config_path=config_path if config_path.exists() else None)
    app_state.mcp_service = service
    await lifecycle.register("mcp_service", service)
    logger.info("MCP module registered", config_path=str(config_path))
