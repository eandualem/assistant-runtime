"""Tests for ToolService lifecycle, not-started guards, and delegation to registry."""

from unittest.mock import MagicMock

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
        assert isinstance(health["backend_tools"], int)

    async def test_start_registers_default_tools(self, service):
        await service.start()
        tool_names = service._registry.get_tool_names()
        # Placeholders
        assert "get_time" in tool_names
        # Agent management tools
        assert "list_agents" in tool_names
        assert "check_agent_state" in tool_names
        assert "start_agent" in tool_names
        assert "send_agent_message" in tool_names
        # Notes management tool
        assert "manage_notes" in tool_names
        # GitHub issue management tools
        assert "create_issue" in tool_names
        assert "search_issues" in tool_names
        assert "get_issue_details" in tool_names
        assert "comment_on_issue" in tool_names
        assert "close_issue" in tool_names
        # Meeting management tools
        assert "create_meeting_room" in tool_names
        assert "list_meeting_rooms" in tool_names
        assert "send_meeting_message" in tool_names
        assert "update_meeting_state" in tool_names
        # Schedule management tools
        assert "add_schedule_item" in tool_names
        assert "remove_schedule_item" in tool_names
        assert "toggle_schedule_item" in tool_names
        # Telegram messaging tools
        assert "respond_telegram" in tool_names
        # Plan management tools
        assert "list_agent_plans" in tool_names
        assert "approve_plan" in tool_names
        assert "reject_plan" in tool_names
        # Artifact management tool
        assert "manage_artifacts" in tool_names


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
        assert (
            result.total_count >= 24
        )  # 1 get_time + 5 agent + 1 notes + 1 artifact + 5 github + 4 meeting + 3 schedule + 1 telegram + 3 plan

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


class TestStateDrivenToolFiltering:
    """Tests for page-based tool filtering via machine_state."""

    async def test_no_machine_state_returns_all(self, service):
        await service.start()
        result = service.get_available_tools(machine_state=None)
        assert result.total_count == 27

    async def test_home_page_returns_all(self, service):
        await service.start()
        state = {"active_page": {"name": "home"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 27

    async def test_agents_page_core_and_agent_and_plan(self, service):
        await service.start()
        state = {"active_page": {"name": "agents"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 15  # 7 core + 5 agent + 3 plan
        names = {t.name for t in result.backend_tools}
        assert "list_agents" in names
        assert "get_time" in names
        assert "manage_notes" in names
        assert "manage_artifacts" in names
        assert "respond_telegram" in names
        assert "approve_plan" in names
        assert "list_agent_plans" in names
        assert "reject_plan" in names
        assert "add_schedule_item" in names

    async def test_sessions_page_core_and_agent_and_plan(self, service):
        await service.start()
        state = {"active_page": {"name": "sessions"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 15  # 7 core + 5 agent + 3 plan
        names = {t.name for t in result.backend_tools}
        assert "start_agent" in names
        assert "approve_plan" in names

    async def test_tasks_page_core_and_github(self, service):
        await service.start()
        state = {"active_page": {"name": "tasks"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 12  # 7 core + 5 github
        names = {t.name for t in result.backend_tools}
        # Core tools present
        assert "get_time" in names
        assert "manage_notes" in names
        assert "add_schedule_item" in names
        # GitHub tools present
        assert "create_issue" in names
        assert "search_issues" in names
        assert "get_issue_details" in names
        assert "comment_on_issue" in names
        assert "close_issue" in names
        # Agent tools excluded
        assert "list_agents" not in names

    async def test_meetings_page_core_and_meeting(self, service):
        await service.start()
        state = {"active_page": {"name": "meetings"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 11  # 7 core + 4 meeting
        names = {t.name for t in result.backend_tools}
        assert "create_meeting_room" in names
        assert "list_meeting_rooms" in names
        assert "send_meeting_message" in names
        assert "update_meeting_state" in names
        assert "get_time" in names
        assert "add_schedule_item" in names
        # Agent tools excluded
        assert "list_agents" not in names

    async def test_flows_page_core_only(self, service):
        await service.start()
        state = {"active_page": {"name": "flows"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 7  # 7 core
        names = {t.name for t in result.backend_tools}
        assert "get_time" in names
        assert "add_schedule_item" in names
        assert "list_agents" not in names

    async def test_repos_page_core_and_repo(self, service):
        await service.start()
        state = {"active_page": {"name": "repos"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 10  # 7 core + 3 repo
        names = {t.name for t in result.backend_tools}
        assert "onboard_repo" in names
        assert "check_repo_status" in names
        assert "list_repos" in names
        assert "get_time" in names
        assert "add_schedule_item" in names
        # Agent tools excluded
        assert "list_agents" not in names

    async def test_unknown_page_returns_all(self, service):
        await service.start()
        state = {"active_page": {"name": "exotic_dashboard"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 27

    async def test_missing_active_page_returns_all(self, service):
        await service.start()
        state = {"some_other_key": "value"}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 27

    async def test_tool_count_warning(self):
        low_max_config = ToolConfig(max_tools_per_request=2)
        svc = ToolService(config=low_max_config)
        await svc.start()
        # meetings page gives tools which exceeds max=2
        state = {"active_page": {"name": "meetings"}}
        result = svc.get_available_tools(machine_state=state)
        assert result.total_count == 11  # still returns them, just warns


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

    async def test_runtime_settings_propagates_to_run_subagent_handler(self):
        """set_runtime_settings updates run_subagent dependencies via public API."""
        mock_llm = MagicMock()
        runtime = MagicMock()
        svc = ToolService(config=ToolConfig(), llm_service=mock_llm)
        await svc.start()

        svc.set_runtime_settings(runtime)

        handler = svc._registry._backend_handlers.get("run_subagent")
        assert handler is not None
        deps = getattr(handler, "_handler_deps", {})
        assert deps.get("runtime_settings") is runtime
