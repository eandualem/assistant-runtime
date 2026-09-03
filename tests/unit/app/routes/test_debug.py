"""Tests for /debug/tools endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.debug import router
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet


def _make_app(tool_service=None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    if tool_service is not None:
        app.state.tool_service = tool_service
    return app


def _make_tool_service(
    backend_names: list[str] | None = None,
    mcp_summary: list[dict] | None = None,
) -> MagicMock:
    svc = MagicMock()

    backend = [
        ToolDefinition(
            name=n, description=f"{n} desc", parameters_schema={}, category=ToolCategory.BACKEND
        )
        for n in (backend_names or [])
    ]
    svc.get_available_tools.return_value = ToolSet(backend_tools=backend)
    svc.build_toolset.return_value = [MagicMock()] * (1 + len(mcp_summary or []))
    svc.get_mcp_summary = AsyncMock(return_value=mcp_summary)
    return svc


class TestDebugToolsEndpoint:
    async def test_returns_backend_tools(self):
        svc = _make_tool_service(backend_names=["get_time", "manage_notes"])
        app = _make_app(svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "get_time" in data["backend_tools"]
        assert "manage_notes" in data["backend_tools"]

    async def test_returns_mcp_section_with_tools(self):
        mcp = [
            {"name": "memory", "tools": ["create_entities", "search_nodes"], "tool_count": 2},
            {"name": "brave-search", "tools": ["brave_web_search"], "tool_count": 1},
        ]
        svc = _make_tool_service(mcp_summary=mcp)
        app = _make_app(svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        data = resp.json()
        assert data["mcp"]["connected"] == 2
        assert len(data["mcp"]["servers"]) == 2
        memory = next(s for s in data["mcp"]["servers"] if s["name"] == "memory")
        assert memory["tool_count"] == 2
        assert "create_entities" in memory["tools"]

    async def test_returns_mcp_prompt_fragment(self):
        mcp = [{"name": "memory", "tools": ["tool_a"], "tool_count": 1}]
        svc = _make_tool_service(mcp_summary=mcp)
        app = _make_app(svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        data = resp.json()
        assert "**memory**" in data["mcp_prompt_fragment"]

    async def test_no_mcp_returns_empty_section(self):
        svc = _make_tool_service()
        app = _make_app(svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        data = resp.json()
        assert data["mcp"]["connected"] == 0
        assert data["mcp"]["servers"] == []
        assert data["mcp_prompt_fragment"] is None

    async def test_toolsets_count(self):
        svc = _make_tool_service()
        svc.build_toolset.return_value = [MagicMock(), MagicMock(), MagicMock()]
        app = _make_app(svc)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        data = resp.json()
        assert data["toolsets_count"] == 3

    async def test_503_when_no_tool_service(self):
        app = _make_app()  # No tool_service on app.state
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/api/debug/tools")
        assert resp.status_code == 503
