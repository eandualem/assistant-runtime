"""Tests for ToolService lifecycle, not-started guards, and delegation to registry."""

import pytest

from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolError
from lovely_assistant.services.tools.interface import ToolService
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


@pytest.fixture
def config():
    return ToolConfig()


@pytest.fixture
def service(config):
    return ToolService(config=config)


class TestLifecycle:
    async def test_start(self, service):
        await service.start()
        assert service._started is True
        health = await service.health_check()
        assert health["healthy"] is True

    async def test_stop(self, service):
        await service.start()
        await service.stop()
        assert service._started is False
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_before_start(self, service):
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_includes_counts(self, service):
        await service.start()
        health = await service.health_check()
        assert "backend_tools" in health
        assert "frontend_tools" in health
        assert isinstance(health["backend_tools"], int)
        assert isinstance(health["frontend_tools"], int)

    async def test_start_registers_placeholders(self, service):
        await service.start()
        tool_names = service._registry.get_tool_names()
        assert "get_time" in tool_names
        assert "ui_notify" in tool_names


class TestNotStartedGuard:
    def test_build_toolset_before_start(self, service):
        with pytest.raises(ToolError, match="not started"):
            service.build_toolset()

    def test_get_available_tools_before_start(self, service):
        with pytest.raises(ToolError, match="not started"):
            service.get_available_tools()

    def test_validate_tool_call_before_start(self, service):
        with pytest.raises(ToolError, match="not started"):
            service.validate_tool_call("anything", {})

    def test_register_backend_before_start(self, service):
        defn = ToolDefinition(
            name="test",
            description="Test",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )

        async def handler() -> str:
            return "test"

        with pytest.raises(ToolError, match="not started"):
            service.register_backend_tool(defn, handler)

    def test_register_frontend_before_start(self, service):
        defn = ToolDefinition(
            name="test",
            description="Test",
            parameters_schema={},
            category=ToolCategory.FRONTEND,
        )
        with pytest.raises(ToolError, match="not started"):
            service.register_frontend_tool(defn)


class TestDelegation:
    async def test_build_toolset(self, service):
        await service.start()
        toolsets = service.build_toolset()
        # Not empty because placeholders are registered (get_time + ui_notify)
        assert isinstance(toolsets, list)
        assert len(toolsets) > 0

    async def test_get_available_tools(self, service):
        await service.start()
        result = service.get_available_tools()
        assert isinstance(result, ToolSet)
        assert result.total_count >= 2  # At least the two placeholders

    async def test_validate_registered_tool(self, service):
        await service.start()
        assert service.validate_tool_call("get_time", {}) is True

    async def test_validate_unknown_tool(self, service):
        await service.start()
        assert service.validate_tool_call("nonexistent", {}) is False

    async def test_register_additional_backend(self, service):
        await service.start()

        async def custom_handler() -> str:
            return "custom"

        defn = ToolDefinition(
            name="custom_backend",
            description="A custom backend tool",
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        )
        service.register_backend_tool(defn, custom_handler)

        tool_names = service._registry.get_tool_names()
        assert "custom_backend" in tool_names

    async def test_register_additional_frontend(self, service):
        await service.start()

        defn = ToolDefinition(
            name="custom_frontend",
            description="A custom frontend tool",
            parameters_schema={
                "type": "object",
                "properties": {"action": {"type": "string"}},
            },
            category=ToolCategory.FRONTEND,
        )
        service.register_frontend_tool(defn)

        tool_names = service._registry.get_tool_names()
        assert "custom_frontend" in tool_names
