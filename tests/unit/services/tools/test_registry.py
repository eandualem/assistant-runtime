"""Tests for ToolRegistry — registration, toolset building, validation, and tool resolution."""

import pytest
from pydantic_ai.toolsets import FunctionToolset

from lovely_assistant.services.tools._registry import ToolRegistry, get_tool_invalidates
from lovely_assistant.services.tools.config import ToolConfig
from lovely_assistant.services.tools.exceptions import ToolValidationError
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition, ToolSet


@pytest.fixture
def config():
    return ToolConfig()


@pytest.fixture
def registry(config):
    return ToolRegistry(config)


@pytest.fixture
def backend_definition():
    return ToolDefinition(
        name="get_time",
        description="Get the current UTC time.",
        parameters_schema={"type": "object", "properties": {}},
        category=ToolCategory.BACKEND,
    )


@pytest.fixture
def dummy_handler():
    async def handler() -> str:
        return "dummy"

    return handler


class TestRegistration:
    def test_register_backend_tool(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        assert "get_time" in registry.get_tool_names()

    def test_duplicate_backend_rejected(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        with pytest.raises(ToolValidationError, match="already registered"):
            registry.register_backend_tool(backend_definition, dummy_handler)


class TestBuildToolset:
    def test_empty_registry(self, registry):
        toolsets = registry.build_toolset()
        assert toolsets == []

    def test_backend_only(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        toolsets = registry.build_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], FunctionToolset)


class TestGetAvailableTools:
    def test_empty(self, registry):
        result = registry.get_available_tools()
        assert isinstance(result, ToolSet)
        assert result.backend_tools == []

    def test_with_tools(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        result = registry.get_available_tools()
        assert len(result.backend_tools) == 1
        assert result.total_count == 1

    def test_returns_toolset_type(self, registry):
        result = registry.get_available_tools()
        assert isinstance(result, ToolSet)


class TestValidateToolCall:
    def test_registered_tool(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        assert registry.validate_tool_call("get_time", {}) is True

    def test_unknown_tool(self, registry):
        assert registry.validate_tool_call("nonexistent", {}) is False


class TestMaxToolsWarning:
    def test_exceeds_max(self):
        """When tool count exceeds max_tools_per_request, build_toolset still works (warning only)."""
        config = ToolConfig(max_tools_per_request=1)
        registry = ToolRegistry(config)

        async def handler_a() -> str:
            return "a"

        async def handler_b() -> str:
            return "b"

        defn_a = ToolDefinition(
            name="tool_a",
            description="Tool A",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        defn_b = ToolDefinition(
            name="tool_b",
            description="Tool B",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )

        registry.register_backend_tool(defn_a, handler_a)
        registry.register_backend_tool(defn_b, handler_b)

        # Should not raise, just warn
        toolsets = registry.build_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], FunctionToolset)


class TestBuildSubagentToolset:
    """Tests for build_subagent_toolset — backend tools minus run_subagent."""

    def test_excludes_run_subagent(self, registry, dummy_handler):
        """run_subagent is excluded from subagent toolsets to prevent recursion."""
        # Register a normal backend tool and run_subagent
        registry.register_backend_tool(
            ToolDefinition(
                name="get_time",
                description="Get time",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            ),
            dummy_handler,
        )

        async def subagent_handler() -> str:
            return "subagent"

        registry.register_backend_tool(
            ToolDefinition(
                name="run_subagent",
                description="Run subagent",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            ),
            subagent_handler,
        )

        toolsets = registry.build_subagent_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], FunctionToolset)

    def test_includes_other_backend_tools(self, registry):
        """Non-subagent backend tools are included in the subagent toolset."""

        async def handler_a() -> str:
            return "a"

        async def handler_b() -> str:
            return "b"

        async def handler_subagent() -> str:
            return "subagent"

        registry.register_backend_tool(
            ToolDefinition(
                name="tool_a",
                description="Tool A",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            ),
            handler_a,
        )
        registry.register_backend_tool(
            ToolDefinition(
                name="tool_b",
                description="Tool B",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            ),
            handler_b,
        )
        registry.register_backend_tool(
            ToolDefinition(
                name="run_subagent",
                description="Run subagent",
                parameters_schema={},
                category=ToolCategory.BACKEND,
            ),
            handler_subagent,
        )

        toolsets = registry.build_subagent_toolset()
        # Should have one FunctionToolset with tool_a and tool_b (not run_subagent)
        assert len(toolsets) == 1

    def test_empty_registry_returns_empty(self, registry):
        """Empty registry produces no toolsets."""
        toolsets = registry.build_subagent_toolset()
        assert toolsets == []


class TestSafetyWrapper:
    """Tests for the _wrap_handler safety net around tool handlers."""

    @pytest.mark.asyncio
    async def test_normal_handler_passes_through(self):
        async def good_handler(x: int) -> str:
            return f"result-{x}"

        wrapped = ToolRegistry._wrap_handler(good_handler, "test_tool")
        result = await wrapped(42)
        assert result == "result-42"

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self):
        async def bad_handler() -> str:
            raise ValueError("kaboom")

        wrapped = ToolRegistry._wrap_handler(bad_handler, "test_tool")
        result = await wrapped()
        assert isinstance(result, dict)
        assert "error" in result
        assert "kaboom" in result["error"]
        assert result["error_code"] == "TOOL_EXECUTION_ERROR"

    def test_preserves_function_name(self):
        async def my_special_tool() -> str:
            return "ok"

        wrapped = ToolRegistry._wrap_handler(my_special_tool, "my_special_tool")
        assert wrapped.__name__ == "my_special_tool"

    @pytest.mark.asyncio
    async def test_retries_on_connection_error_then_succeeds(self):
        """Wrapped handler retries on ConnectionError and returns success."""
        call_count = 0

        async def flaky_handler() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("transient")
            return "recovered"

        wrapped = ToolRegistry._wrap_handler(flaky_handler, "flaky_tool")
        result = await wrapped()
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_gives_up_after_two_attempts_returns_error_dict(self):
        """Wrapped handler returns error dict after exhausting retry attempts."""
        call_count = 0

        async def always_timeout() -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("always times out")

        wrapped = ToolRegistry._wrap_handler(always_timeout, "timeout_tool")
        result = await wrapped()
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error_code"] == "TOOL_EXECUTION_ERROR"
        # 2 attempts total (1 initial + 1 retry)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_on_value_error(self):
        """Non-transient errors like ValueError are not retried."""
        call_count = 0

        async def bad_input() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("bad args")

        wrapped = ToolRegistry._wrap_handler(bad_input, "bad_tool")
        result = await wrapped()
        assert isinstance(result, dict)
        assert "bad args" in result["error"]
        # Should be called only once — no retries for ValueError
        assert call_count == 1


class TestToolInvalidates:
    def test_known_tool_returns_list(self):
        result = get_tool_invalidates("create_issue")
        assert result == ["tasks"]

    def test_unknown_tool_returns_none(self):
        result = get_tool_invalidates("get_time")
        assert result is None

    def test_agent_tools_invalidate_agents(self):
        result = get_tool_invalidates("start_agent")
        assert result == ["agents"]

    def test_meeting_tools_invalidate_meetings(self):
        result = get_tool_invalidates("create_meeting_room")
        assert result == ["meetings"]


class TestFrontendToolRegistration:
    def test_register_frontend_tools(self, registry):
        registry.register_frontend_tools()
        assert registry.frontend_tool_count() == 2

    def test_frontend_tools_in_available_tools(self, registry):
        registry.register_frontend_tools()
        result = registry.get_available_tools()
        assert len(result.frontend_tools) == 2

    def test_frontend_tools_bypass_page_filter(self, registry):
        registry.register_frontend_tools()
        machine_state = {"active_page": {"name": "flows"}}
        result = registry.get_available_tools(machine_state=machine_state)
        assert len(result.frontend_tools) == 2
