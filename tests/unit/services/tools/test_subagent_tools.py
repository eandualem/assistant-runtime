"""Tests for subagent definitions, registry, and tool registration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._subagent_tools import (
    SUBAGENT_REGISTRY,
    SubagentDefinition,
    get_subagent,
    list_subagents,
    register_subagent_tools,
)
from lovely_assistant.services.tools.config import ToolConfig

# ---------------------------------------------------------------------------
# Module path for patching
# ---------------------------------------------------------------------------

MODULE = "lovely_assistant.services.tools._subagent_tools"


# ---------------------------------------------------------------------------
# TestSubagentDefinition
# ---------------------------------------------------------------------------


class TestSubagentDefinition:
    def test_default_values(self):
        defn = SubagentDefinition(
            id="test",
            name="Test Agent",
            description="A test subagent",
            system_prompt="You are a test agent.",
        )
        assert defn.id == "test"
        assert defn.name == "Test Agent"
        assert defn.description == "A test subagent"
        assert defn.system_prompt == "You are a test agent."
        assert defn.default_model is None
        assert defn.default_thinking_budget is None
        assert defn.max_iterations == 10

    def test_custom_values(self):
        defn = SubagentDefinition(
            id="custom",
            name="Custom Agent",
            description="A custom subagent",
            system_prompt="You are custom.",
            default_model="openai:gpt-4o",
            default_thinking_budget=5000,
            max_iterations=25,
        )
        assert defn.id == "custom"
        assert defn.name == "Custom Agent"
        assert defn.description == "A custom subagent"
        assert defn.system_prompt == "You are custom."
        assert defn.default_model == "openai:gpt-4o"
        assert defn.default_thinking_budget == 5000
        assert defn.max_iterations == 25


# ---------------------------------------------------------------------------
# TestSubagentRegistry
# ---------------------------------------------------------------------------


class TestSubagentRegistry:
    def test_get_known_subagent(self):
        result = get_subagent("researcher")
        assert result is not None
        assert isinstance(result, SubagentDefinition)
        assert result.id == "researcher"
        assert result.name == "Research Agent"

    def test_get_unknown_returns_none(self):
        result = get_subagent("nonexistent")
        assert result is None

    def test_list_subagents_returns_summaries(self):
        result = list_subagents()
        assert isinstance(result, list)
        assert len(result) >= 1

        # Each entry should have id, name, description keys
        for entry in result:
            assert "id" in entry
            assert "name" in entry
            assert "description" in entry

        # Researcher should be in the list
        ids = [s["id"] for s in result]
        assert "researcher" in ids

    def test_list_subagents_does_not_expose_system_prompt(self):
        result = list_subagents()
        for entry in result:
            assert "system_prompt" not in entry


# ---------------------------------------------------------------------------
# TestRunSubagentTool
# ---------------------------------------------------------------------------


class TestRunSubagentTool:
    async def test_unknown_subagent_returns_error(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)

        handler = registry._backend_handlers["run_subagent"]

        # Create a mock RunContext
        mock_ctx = MagicMock()
        mock_ctx.usage = MagicMock()

        result = await handler(mock_ctx, task="do something", subagent_id="nonexistent")
        assert isinstance(result, dict)
        assert result["error_code"] == "SUBAGENT_NOT_FOUND"
        assert "nonexistent" in result["error"]
        assert "available_subagents" in result
        assert "researcher" in result["available_subagents"]

    def test_default_subagent_is_researcher(self):
        assert "researcher" in SUBAGENT_REGISTRY
        # Verify researcher is the only initially registered subagent
        assert len(SUBAGENT_REGISTRY) == 1


# ---------------------------------------------------------------------------
# TestRegisterSubagentTools
# ---------------------------------------------------------------------------


class TestRegisterSubagentTools:
    def test_tool_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)

        names = registry.get_tool_names()
        assert "run_subagent" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        assert len(registry._backend_definitions) == 1

    def test_is_backend_tool(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        defn = registry._backend_definitions["run_subagent"]
        assert defn.category == "backend"

    def test_definition_has_schema(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        defn = registry._backend_definitions["run_subagent"]
        assert isinstance(defn.parameters_schema, dict)
        assert defn.description
        assert "task" in defn.parameters_schema["properties"]

    def test_handler_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        assert "run_subagent" in registry._backend_handlers
        assert callable(registry._backend_handlers["run_subagent"])

    def test_task_is_required(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        schema = registry._backend_definitions["run_subagent"].parameters_schema
        required = schema.get("required", [])
        assert "task" in required

    def test_subagent_id_not_required(self):
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        schema = registry._backend_definitions["run_subagent"].parameters_schema
        required = schema.get("required", [])
        assert "subagent_id" not in required


# ---------------------------------------------------------------------------
# TestRunSubagentRuntimeSettings
# ---------------------------------------------------------------------------


class TestRunSubagentRuntimeSettings:
    """Tests for runtime settings being passed through to execute_subagent."""

    def _setup_handler(self):
        """Register tools and return the run_subagent handler."""
        registry = ToolRegistry(ToolConfig())
        register_subagent_tools(registry)
        return registry._backend_handlers["run_subagent"]

    def _make_mock_ctx(self):
        """Create a mock RunContext with usage."""
        mock_ctx = MagicMock()
        mock_ctx.usage = MagicMock()
        return mock_ctx

    @patch(
        "lovely_assistant.services.tools._subagent_executor.execute_subagent",
        new_callable=AsyncMock,
    )
    async def test_runtime_model_passed_to_executor(self, mock_execute):
        mock_execute.return_value = {"result": "done", "_metadata": {}}

        handler = self._setup_handler()
        mock_ctx = self._make_mock_ctx()

        runtime = MagicMock()
        runtime.get = MagicMock(
            side_effect=lambda k, default=None: {
                "subagent_model": "openai:gpt-4o",
            }.get(k, default)
        )

        handler._subagent_deps = {
            "llm_service": MagicMock(),
            "get_backend_toolsets": MagicMock(return_value=[]),
            "runtime_settings": runtime,
        }

        await handler(mock_ctx, task="research something")
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args.kwargs
        assert call_kwargs["model_override"] == "openai:gpt-4o"

    @patch(
        "lovely_assistant.services.tools._subagent_executor.execute_subagent",
        new_callable=AsyncMock,
    )
    async def test_runtime_thinking_budget_passed_to_executor(self, mock_execute):
        mock_execute.return_value = {"result": "done", "_metadata": {}}

        handler = self._setup_handler()
        mock_ctx = self._make_mock_ctx()

        runtime = MagicMock()
        runtime.get = MagicMock(
            side_effect=lambda k, default=None: {
                "subagent_thinking_budget": 5000,
            }.get(k, default)
        )

        handler._subagent_deps = {
            "llm_service": MagicMock(),
            "get_backend_toolsets": MagicMock(return_value=[]),
            "runtime_settings": runtime,
        }

        await handler(mock_ctx, task="analyze data")
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args.kwargs
        assert call_kwargs["thinking_budget_override"] == 5000

    @patch(
        "lovely_assistant.services.tools._subagent_executor.execute_subagent",
        new_callable=AsyncMock,
    )
    async def test_no_runtime_settings_passes_none(self, mock_execute):
        mock_execute.return_value = {"result": "done", "_metadata": {}}

        handler = self._setup_handler()
        mock_ctx = self._make_mock_ctx()

        handler._subagent_deps = {
            "llm_service": MagicMock(),
            "get_backend_toolsets": MagicMock(return_value=[]),
        }

        await handler(mock_ctx, task="check status")
        mock_execute.assert_called_once()
        call_kwargs = mock_execute.call_args.kwargs
        assert call_kwargs["model_override"] is None
        assert call_kwargs["thinking_budget_override"] is None
