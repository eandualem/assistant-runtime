"""Tests for ToolService lifecycle, not-started guards, and delegation to registry."""

from unittest.mock import MagicMock

import pytest

from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolError
from assistant_runtime.services.tools.interface import ToolService
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet


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
        assert isinstance(health["backend_tools"], int)

    async def test_start_registers_default_tools(self, service):
        await service.start()
        tool_names = service._registry.get_tool_names()
        # Built in
        assert "get_time" in tool_names
        assert "look_at_screen" in tool_names
        assert "manage_artifacts" in tool_names
        # Capabilities need a configured provider
        for name in ("manage_notes", "list_documents", "list_agents", "create_meeting_room"):
            assert name not in tool_names
        # Integrations still on the old model (fail soft when unconfigured)
        assert "create_issue" in tool_names
        assert "respond_telegram" in tool_names

    def test_build_toolset_before_start(self, service):
        with pytest.raises(ToolError, match="not started"):
            service.build_toolset()

    def test_get_available_tools_before_start(self, service):
        with pytest.raises(ToolError, match="not started"):
            service.get_available_tools()

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


class TestDelegation:
    async def test_build_toolset(self, service):
        await service.start()
        toolsets = service.build_toolset()
        # Not empty because default tools are registered (get_time, agent tools, etc.)
        assert isinstance(toolsets, list)
        assert len(toolsets) > 0

    async def test_get_available_tools(self, service):
        await service.start()
        result = service.get_available_tools()
        assert isinstance(result, ToolSet)
        assert result.total_count >= 3

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


class TestHostContextToolScoping:
    """Page scoping comes from ToolConfig.page_scopes; nothing is scoped by default."""

    async def test_no_context_returns_all(self, service):
        await service.start()
        result = service.get_available_tools(host_context=None)
        assert result.total_count == service._registry.backend_tool_count()
        assert result.host_tools == []

    async def test_any_page_returns_all_without_scopes(self, service):
        await service.start()
        everything = service._registry.backend_tool_count()
        for page in ("home", "tasks", "flows", "exotic"):
            result = service.get_available_tools(host_context={"page": {"name": page}})
            assert result.total_count == everything

    async def test_context_without_page_returns_all(self, service):
        await service.start()
        result = service.get_available_tools(host_context={"some_other_key": "value"})
        assert result.total_count == service._registry.backend_tool_count()

    async def test_configured_scope_limits_backend_tools(self):
        svc = ToolService(config=ToolConfig(page_scopes={"tasks": ["create_issue", "get_time"]}))
        await svc.start()
        result = svc.get_available_tools(host_context={"page": {"name": "tasks"}})
        assert {t.name for t in result.backend_tools} == {"create_issue", "get_time"}
        assert result.filtered_out_count == svc._registry.backend_tool_count() - 2

    async def test_scope_names_not_registered_are_ignored(self):
        svc = ToolService(config=ToolConfig(page_scopes={"x": ["no_such_tool", "get_time"]}))
        await svc.start()
        result = svc.get_available_tools(host_context={"page": {"name": "x"}})
        assert [t.name for t in result.backend_tools] == ["get_time"]

    async def test_host_tools_from_config(self):
        svc = ToolService(
            config=ToolConfig(
                host_tools={"navigate": {"description": "Go", "parameters": {"type": "object"}}}
            )
        )
        await svc.start()
        assert svc.is_host_tool("navigate")
        assert not svc.is_host_tool("get_time")
        assert [t.name for t in svc.get_available_tools().host_tools] == ["navigate"]
        health = await svc.health_check()
        assert health["host_tools"] == 1

    async def test_invalidations_from_config(self):
        svc = ToolService(config=ToolConfig(invalidations={"create_issue": ["tasks"]}))
        await svc.start()
        assert svc.get_tool_invalidates("create_issue") == ["tasks"]
        assert svc.get_tool_invalidates("get_time") is None

    async def test_warm_host_context_requires_started(self, service):
        with pytest.raises(ToolError):
            service.warm_host_context({"page": {"name": "tasks"}})

    async def test_tool_count_warning(self):
        low_max_config = ToolConfig(max_tools_per_request=2)
        svc = ToolService(config=low_max_config)
        await svc.start()
        result = svc.get_available_tools(host_context={"page": {"name": "meetings"}})
        assert result.total_count == svc._registry.backend_tool_count()


class TestSubagentIntegration:
    """Tests for subagent tool registration when llm_service is provided."""

    async def test_run_subagent_registered_with_llm_service(self):
        """When llm_service is provided, run_subagent tool is registered."""
        mock_llm = MagicMock()
        svc = ToolService(config=ToolConfig(), llm_service=mock_llm)
        await svc.start()

        tool_names = svc._registry.get_tool_names()
        assert "run_subagent" in tool_names

    async def test_run_subagent_not_registered_without_llm_service(self):
        """Without llm_service, run_subagent is NOT registered."""
        svc = ToolService(config=ToolConfig())
        await svc.start()

        tool_names = svc._registry.get_tool_names()
        assert "run_subagent" not in tool_names

    async def test_tool_count_with_llm_service(self):
        """With llm_service, total tool count increases by 1."""
        svc_without = ToolService(config=ToolConfig())
        await svc_without.start()
        count_without = svc_without.get_available_tools().total_count

        mock_llm = MagicMock()
        svc_with = ToolService(config=ToolConfig(), llm_service=mock_llm)
        await svc_with.start()
        count_with = svc_with.get_available_tools().total_count

        assert count_with == count_without + 1

    async def test_get_subagent_toolsets(self):
        """get_subagent_toolsets returns toolsets without run_subagent."""
        mock_llm = MagicMock()
        svc = ToolService(config=ToolConfig(), llm_service=mock_llm)
        await svc.start()

        toolsets = svc.get_subagent_toolsets()
        assert isinstance(toolsets, list)

    async def test_runtime_settings_reach_run_subagent(self):
        """Settings attached after start are read by the subagent tool on each call."""
        from unittest.mock import AsyncMock, patch

        svc = ToolService(config=ToolConfig(), llm_service=MagicMock())
        await svc.start()
        runtime = MagicMock()
        runtime.get = MagicMock(
            side_effect=lambda k, default=None: {"subagent_model": "openai:gpt-5.4"}.get(k, default)
        )
        svc.set_runtime_settings(runtime)

        handler = svc._registry._backend_handlers["run_subagent"]
        with patch(
            "assistant_runtime.services.tools.builtin._subagent_executor.execute_subagent",
            new_callable=AsyncMock,
            return_value={"result": "done"},
        ) as execute:
            await handler(MagicMock(), task="look")

        assert execute.call_args.kwargs["model_override"] == "openai:gpt-5.4"
