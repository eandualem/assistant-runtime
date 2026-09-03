"""Tests for MCPService lifecycle, toolset provision, and health reporting."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from assistant_runtime.services.mcp.interface import MCPService


def _make_mock_tool(name: str) -> MagicMock:
    """Create a mock MCP Tool with a name attribute."""
    tool = MagicMock()
    tool.name = name
    return tool


def _make_mock_server(
    name: str, fail_on_enter: bool = False, tool_names: list[str] | None = None
) -> MagicMock:
    """Create a mock MCP server that acts as an async context manager."""
    server = MagicMock()
    server.id = name
    server.tool_prefix = name

    if fail_on_enter:

        async def fail_enter(*args, **kwargs):
            raise ConnectionError(f"Failed to connect to {name}")

        server.__aenter__ = AsyncMock(side_effect=fail_enter)
    else:
        server.__aenter__ = AsyncMock(return_value=server)
    server.__aexit__ = AsyncMock(return_value=None)

    # Mock list_tools
    tools = [_make_mock_tool(t) for t in (tool_names or [])]
    server.list_tools = AsyncMock(return_value=tools)

    return server


class TestStartNoConfig:
    async def test_starts_with_no_config(self):
        service = MCPService(config_path=None)
        await service.start()
        assert service._started is True

    async def test_no_config_returns_empty_toolsets(self):
        service = MCPService(config_path=None)
        await service.start()
        assert service.get_toolsets() == []

    async def test_no_config_health_is_healthy(self):
        service = MCPService(config_path=None)
        await service.start()
        health = await service.health_check()
        assert health["healthy"] is True
        assert health["connected"] == 0


class TestStartLoadsServers:
    async def test_loads_and_connects_servers(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server1 = _make_mock_server("memory")
        server2 = _make_mock_server("brave-search")

        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server1, server2],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        assert service._started is True
        assert len(service._live_servers) == 2
        server1.__aenter__.assert_awaited_once()
        server2.__aenter__.assert_awaited_once()

    async def test_servers_returned_as_toolsets(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory")
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        toolsets = service.get_toolsets()
        assert len(toolsets) == 1
        assert toolsets[0] is server


class TestGracefulServerFailure:
    async def test_one_fails_others_succeed(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        good_server = _make_mock_server("memory")
        bad_server = _make_mock_server("broken", fail_on_enter=True)

        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[good_server, bad_server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        assert len(service._live_servers) == 1
        assert len(service._failed) == 1
        assert service._failed[0]["name"] == "broken"

    async def test_all_fail_still_starts(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        bad1 = _make_mock_server("bad1", fail_on_enter=True)
        bad2 = _make_mock_server("bad2", fail_on_enter=True)

        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[bad1, bad2],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        assert service._started is True
        assert len(service._live_servers) == 0
        assert len(service._failed) == 2

    async def test_config_load_error_handled_gracefully(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            side_effect=ValueError("bad config"),
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        assert service._started is True
        assert service.get_toolsets() == []


class TestStop:
    async def test_stop_clears_state(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory")
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()
            assert len(service._live_servers) == 1

            await service.stop()

        assert service._started is False
        assert service._live_servers == []
        assert service._exit_stack is None

    async def test_stop_without_start(self):
        service = MCPService(config_path=None)
        await service.stop()  # Should not raise
        assert service._started is False


class TestHealthCheck:
    async def test_before_start(self):
        service = MCPService(config_path=None)
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_after_start_with_servers(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory")
        bad = _make_mock_server("broken", fail_on_enter=True)
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server, bad],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        health = await service.health_check()
        assert health["healthy"] is True
        assert health["connected"] == 1
        assert len(health["failed"]) == 1
        assert "memory" in health["server_names"]


class TestGetServerSummary:
    async def test_empty_when_no_servers(self):
        service = MCPService(config_path=None)
        await service.start()
        assert service.get_server_summary() == []

    async def test_summary_contains_server_names(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server1 = _make_mock_server("memory")
        server2 = _make_mock_server("brave-search")
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server1, server2],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        summary = service.get_server_summary()
        assert len(summary) == 2
        names = [s["name"] for s in summary]
        assert "memory" in names
        assert "brave-search" in names


class TestGetDetailedSummary:
    async def test_empty_when_no_servers(self):
        service = MCPService(config_path=None)
        await service.start()
        result = await service.get_detailed_summary()
        assert result == []

    async def test_returns_tool_names_and_counts(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server1 = _make_mock_server(
            "memory", tool_names=["create_entities", "add_observations", "search_nodes"]
        )
        server2 = _make_mock_server("brave-search", tool_names=["brave_web_search"])
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server1, server2],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        result = await service.get_detailed_summary()
        assert len(result) == 2

        memory_entry = next(s for s in result if s["name"] == "memory")
        assert memory_entry["tool_count"] == 3
        assert "create_entities" in memory_entry["tools"]

        brave_entry = next(s for s in result if s["name"] == "brave-search")
        assert brave_entry["tool_count"] == 1
        assert brave_entry["tools"] == ["brave_web_search"]

    async def test_caches_after_first_call(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory", tool_names=["tool_a"])
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        await service.get_detailed_summary()
        await service.get_detailed_summary()

        # list_tools called only once (cached)
        server.list_tools.assert_awaited_once()

    async def test_handles_list_tools_failure_gracefully(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory")
        server.list_tools = AsyncMock(side_effect=RuntimeError("connection lost"))
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        result = await service.get_detailed_summary()
        assert len(result) == 1
        assert result[0]["name"] == "memory"
        assert result[0]["tools"] == []
        assert result[0]["tool_count"] == 0

    async def test_stop_clears_cache(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        server = _make_mock_server("memory", tool_names=["tool_a"])
        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[server],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        await service.get_detailed_summary()
        assert service._detailed_cache is not None

        await service.stop()
        assert service._detailed_cache is None


class TestEmptyServerList:
    async def test_empty_server_list(self, tmp_path):
        config_file = tmp_path / "mcp.json"
        config_file.write_text('{"mcpServers": {}}')

        with patch(
            "assistant_runtime.services.mcp.interface.load_mcp_toolsets",
            return_value=[],
        ):
            service = MCPService(config_path=config_file)
            await service.start()

        assert service._started is True
        assert service.get_toolsets() == []
        assert service._exit_stack is None
