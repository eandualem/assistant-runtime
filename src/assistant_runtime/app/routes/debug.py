"""Debug routes — runtime diagnostics for tool and MCP inspection."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from assistant_runtime.app.assistant._prompt_builder import _mcp_connections_fragment

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/tools")
async def debug_tools(request: Request) -> dict[str, Any]:
    """Return complete tool registry and MCP server status as seen by the runtime."""
    tool_service = getattr(request.app.state, "tool_service", None)
    if tool_service is None:
        return JSONResponse(status_code=503, content={"detail": "Tool service not available"})

    # Same code path as chat flow: get_available_tools(None) = all tools, no page filter
    available = tool_service.get_available_tools(None)

    # Same code path as chat flow: build_toolset(None)
    toolsets = tool_service.build_toolset(None)

    # Same code path as chat flow: get_mcp_summary()
    mcp_summary = await tool_service.get_mcp_summary()

    # Same code path as chat flow: _mcp_connections_fragment()
    mcp_prompt_fragment = _mcp_connections_fragment(mcp_summary)

    # Build MCP section
    mcp_section: dict[str, Any] = {"connected": 0, "servers": []}
    if mcp_summary:
        mcp_section["connected"] = len(mcp_summary)
        for server in mcp_summary:
            mcp_section["servers"].append(
                {
                    "name": server.get("name", "unknown"),
                    "is_running": True,
                    "tools": server.get("tools", []),
                    "tool_count": server.get("tool_count", 0),
                }
            )

    return {
        "backend_tools": [t.name for t in available.backend_tools],
        "mcp": mcp_section,
        "toolsets_count": len(toolsets),
        "mcp_prompt_fragment": mcp_prompt_fragment or None,
    }
