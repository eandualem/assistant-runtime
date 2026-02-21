"""Tests for ToolRegistry — registration, toolset building, validation, and tool resolution."""

import pytest
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

from lovely_assistant.services.tools._registry import ToolRegistry
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
def frontend_definition():
    return ToolDefinition(
        name="navigate",
        description="Navigate to a page.",
        parameters_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        category=ToolCategory.FRONTEND,
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

    def test_register_frontend_tool(self, registry, frontend_definition):
        registry.register_frontend_tool(frontend_definition)
        assert "navigate" in registry.get_tool_names()

    def test_duplicate_backend_rejected(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        with pytest.raises(ToolValidationError, match="already registered"):
            registry.register_backend_tool(backend_definition, dummy_handler)

    def test_duplicate_frontend_rejected(self, registry, frontend_definition):
        registry.register_frontend_tool(frontend_definition)
        with pytest.raises(ToolValidationError, match="already registered"):
            registry.register_frontend_tool(frontend_definition)

    def test_wrong_category_backend(self, registry, frontend_definition, dummy_handler):
        """Registering a frontend-categorized definition as a backend tool raises."""
        with pytest.raises(ToolValidationError, match="Expected backend tool"):
            registry.register_backend_tool(frontend_definition, dummy_handler)

    def test_wrong_category_frontend(self, registry, backend_definition):
        """Registering a backend-categorized definition as a frontend tool raises."""
        with pytest.raises(ToolValidationError, match="Expected frontend tool"):
            registry.register_frontend_tool(backend_definition)


class TestBuildToolset:
    def test_empty_registry(self, registry):
        toolsets = registry.build_toolset()
        assert toolsets == []

    def test_backend_only(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        toolsets = registry.build_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], FunctionToolset)

    def test_frontend_only(self, registry, frontend_definition):
        registry.register_frontend_tool(frontend_definition)
        toolsets = registry.build_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], ExternalToolset)

    def test_both(self, registry, backend_definition, frontend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        registry.register_frontend_tool(frontend_definition)
        toolsets = registry.build_toolset()
        assert len(toolsets) == 2
        types = {type(t) for t in toolsets}
        assert FunctionToolset in types
        assert ExternalToolset in types

    def test_frontend_disabled(self, backend_definition, frontend_definition, dummy_handler):
        """With enable_frontend_tools=False, only backend toolset is returned."""
        config = ToolConfig(enable_frontend_tools=False)
        registry = ToolRegistry(config)
        registry.register_backend_tool(backend_definition, dummy_handler)
        registry.register_frontend_tool(frontend_definition)

        toolsets = registry.build_toolset()
        assert len(toolsets) == 1
        assert isinstance(toolsets[0], FunctionToolset)


class TestGetAvailableTools:
    def test_empty(self, registry):
        result = registry.get_available_tools()
        assert isinstance(result, ToolSet)
        assert result.backend_tools == []
        assert result.frontend_tools == []

    def test_with_tools(self, registry, backend_definition, frontend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        registry.register_frontend_tool(frontend_definition)
        result = registry.get_available_tools()
        assert len(result.backend_tools) == 1
        assert len(result.frontend_tools) == 1
        assert result.total_count == 2

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
