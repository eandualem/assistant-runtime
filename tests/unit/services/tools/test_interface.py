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

    async def test_start_registers_default_tools(self, service):
        await service.start()
        tool_names = service._registry.get_tool_names()
        # Placeholders
        assert "get_time" in tool_names
        assert "ui_notify" in tool_names
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
        # Plan management tools
        assert "list_agent_plans" in tool_names
        assert "approve_plan" in tool_names
        assert "reject_plan" in tool_names
        # Navigation frontend tool
        assert "navigate" in tool_names


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
        assert (
            result.total_count >= 23
        )  # 2 placeholders + 4 agent + 1 notes + 5 github + 4 meeting + 3 schedule + 3 plan

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


class TestStateDrivenToolFiltering:
    """Tests for page-based tool filtering via machine_state."""

    async def test_no_machine_state_returns_all(self, service):
        await service.start()
        result = service.get_available_tools(machine_state=None)
        assert result.total_count == 23

    async def test_home_page_returns_all(self, service):
        await service.start()
        state = {"active_page": {"name": "home"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 23

    async def test_agents_page_core_and_agent_and_plan(self, service):
        await service.start()
        state = {"active_page": {"name": "agents"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 14  # 7 core + 4 agent + 3 plan
        names = {t.name for t in result.backend_tools} | {t.name for t in result.frontend_tools}
        assert "list_agents" in names
        assert "get_time" in names
        assert "manage_notes" in names
        assert "approve_plan" in names
        assert "list_agent_plans" in names
        assert "reject_plan" in names
        assert "add_schedule_item" in names

    async def test_sessions_page_core_and_agent_and_plan(self, service):
        await service.start()
        state = {"active_page": {"name": "sessions"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 14  # 7 core + 4 agent + 3 plan
        names = {t.name for t in result.backend_tools} | {t.name for t in result.frontend_tools}
        assert "start_agent" in names
        assert "ui_notify" in names
        assert "approve_plan" in names

    async def test_tasks_page_core_and_github(self, service):
        await service.start()
        state = {"active_page": {"name": "tasks"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 12  # 7 core + 5 github
        names = {t.name for t in result.backend_tools} | {t.name for t in result.frontend_tools}
        # Core tools present
        assert "get_time" in names
        assert "manage_notes" in names
        assert "ui_notify" in names
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
        names = {t.name for t in result.backend_tools} | {t.name for t in result.frontend_tools}
        assert "create_meeting_room" in names
        assert "list_meeting_rooms" in names
        assert "send_meeting_message" in names
        assert "update_meeting_state" in names
        assert "get_time" in names
        assert "navigate" in names
        assert "add_schedule_item" in names
        # Agent tools excluded
        assert "list_agents" not in names

    async def test_flows_page_core_only(self, service):
        await service.start()
        state = {"active_page": {"name": "flows"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 7  # 7 core
        names = {t.name for t in result.backend_tools} | {t.name for t in result.frontend_tools}
        assert "get_time" in names
        assert "add_schedule_item" in names
        assert "list_agents" not in names

    async def test_unknown_page_returns_all(self, service):
        await service.start()
        state = {"active_page": {"name": "exotic_dashboard"}}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 23

    async def test_missing_active_page_returns_all(self, service):
        await service.start()
        state = {"some_other_key": "value"}
        result = service.get_available_tools(machine_state=state)
        assert result.total_count == 23

    async def test_tool_count_warning(self):
        low_max_config = ToolConfig(max_tools_per_request=2)
        svc = ToolService(config=low_max_config)
        await svc.start()
        # meetings page gives 10 tools which exceeds max=2
        state = {"active_page": {"name": "meetings"}}
        result = svc.get_available_tools(machine_state=state)
        assert result.total_count == 11  # still returns them, just warns
