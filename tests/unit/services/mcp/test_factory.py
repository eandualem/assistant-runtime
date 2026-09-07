"""Tests for MCP service factory registration."""

from unittest.mock import AsyncMock, MagicMock

from assistant_runtime.services.mcp.factory import DEFAULT_CONFIG_NAME, register_mcp
from assistant_runtime.services.mcp.interface import MCPService


class TestRegisterMcp:
    async def test_stores_service_on_app_state(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)  # no mcp_servers.json here
        app_state = MagicMock()
        lifecycle = AsyncMock()

        await register_mcp(app_state, lifecycle)

        assert hasattr(app_state, "mcp_service")
        assert isinstance(app_state.mcp_service, MCPService)

    async def test_registers_with_lifecycle(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        app_state = MagicMock()
        lifecycle = AsyncMock()

        await register_mcp(app_state, lifecycle)

        lifecycle.register.assert_awaited_once()
        call_args = lifecycle.register.call_args
        assert call_args[0][0] == "mcp_service"
        assert isinstance(call_args[0][1], MCPService)

    async def test_uses_env_var_config_path(self, tmp_path, monkeypatch):
        config_file = tmp_path / "custom_mcp.json"
        config_file.write_text('{"mcpServers": {}}')
        monkeypatch.setenv("MCP_CONFIG_PATH", str(config_file))

        app_state = MagicMock()
        lifecycle = AsyncMock()
        await register_mcp(app_state, lifecycle)

        service = app_state.mcp_service
        assert service._config_path == config_file

    async def test_no_config_file_sets_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        app_state = MagicMock()
        lifecycle = AsyncMock()

        await register_mcp(app_state, lifecycle)

        assert app_state.mcp_service._config_path is None

    async def test_config_in_working_directory_is_used(self, tmp_path, monkeypatch):
        """Installed-package layout: the file is found next to where the process runs."""
        monkeypatch.delenv("MCP_CONFIG_PATH", raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / DEFAULT_CONFIG_NAME
        config_file.write_text('{"mcpServers": {}}')
        app_state = MagicMock()
        lifecycle = AsyncMock()

        await register_mcp(app_state, lifecycle)

        assert app_state.mcp_service._config_path == config_file
