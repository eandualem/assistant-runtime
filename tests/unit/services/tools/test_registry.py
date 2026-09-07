"""Tests for ToolRegistry — registration, toolset building, validation, and tool resolution."""

import pytest
from pydantic_ai.toolsets import FunctionToolset

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolValidationError
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition, ToolSet


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


NAVIGATE = {
    "navigate": {
        "description": "Navigate the host to a page.",
        "parameters": {"type": "object", "properties": {"page": {"type": "string"}}},
    }
}


class TestToolInvalidates:
    def test_configured_tool_returns_domains(self):
        registry = ToolRegistry(ToolConfig(invalidations={"create_issue": ["tasks"]}))
        assert registry.invalidates_for("create_issue") == ["tasks"]

    def test_unconfigured_tool_returns_none(self, registry):
        assert registry.invalidates_for("create_issue") is None

    def test_empty_list_returns_none(self):
        registry = ToolRegistry(ToolConfig(invalidations={"x": []}))
        assert registry.invalidates_for("x") is None


class TestHostToolRegistration:
    def test_none_by_default(self, registry):
        registry.register_host_tools()
        assert registry.host_tool_count() == 0
        assert registry.build_toolset() == []

    def test_from_config(self):
        registry = ToolRegistry(ToolConfig(host_tools=NAVIGATE))
        registry.register_host_tools()
        assert registry.host_tool_count() == 1
        assert registry.is_host_tool("navigate")
        assert not registry.is_host_tool("get_time")

    def test_explicit_schemas(self, registry):
        registry.register_host_tools(NAVIGATE)
        assert [d.name for d in registry.get_available_tools().host_tools] == ["navigate"]

    def test_host_tools_bypass_page_scope(self):
        registry = ToolRegistry(ToolConfig(host_tools=NAVIGATE, page_scopes={"flows": []}))
        registry.register_host_tools()
        result = registry.get_available_tools({"page": {"name": "flows"}})
        assert len(result.host_tools) == 1

    def test_clash_with_backend_tool_rejected(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        with pytest.raises(ToolValidationError):
            registry.register_host_tools({"get_time": NAVIGATE["navigate"]})

    def test_backend_tool_clashing_with_host_tool_rejected(self, backend_definition, dummy_handler):
        registry = ToolRegistry(ToolConfig(host_tools={"get_time": NAVIGATE["navigate"]}))
        registry.register_host_tools()
        with pytest.raises(ToolValidationError):
            registry.register_backend_tool(backend_definition, dummy_handler)

    def test_toolset_includes_external_toolset(self):
        from pydantic_ai.toolsets import ExternalToolset

        registry = ToolRegistry(ToolConfig(host_tools=NAVIGATE))
        registry.register_host_tools()
        assert any(isinstance(t, ExternalToolset) for t in registry.build_toolset())


class TestPageScopes:
    @pytest.fixture
    def scoped(self, dummy_handler):
        registry = ToolRegistry(ToolConfig(page_scopes={"tasks": ["a"], "empty": []}))
        for name in ("a", "b"):
            registry.register_backend_tool(
                ToolDefinition(
                    name=name, description=name, parameters_schema={}, category=ToolCategory.BACKEND
                ),
                dummy_handler,
            )
        return registry

    def test_no_context_all_tools(self, scoped):
        assert {t.name for t in scoped.get_available_tools(None).backend_tools} == {"a", "b"}

    def test_unlisted_page_all_tools(self, scoped):
        result = scoped.get_available_tools({"page": {"name": "home"}})
        assert {t.name for t in result.backend_tools} == {"a", "b"}
        assert result.filtered_out_count == 0

    def test_listed_page_scoped(self, scoped):
        result = scoped.get_available_tools({"page": {"name": "tasks"}})
        assert [t.name for t in result.backend_tools] == ["a"]
        assert result.filtered_out_count == 1
        assert result.page == "tasks"

    def test_empty_scope_hides_everything(self, scoped):
        assert scoped.get_available_tools({"page": {"name": "empty"}}).backend_tools == []

    def test_context_without_page_all_tools(self, scoped):
        result = scoped.get_available_tools({"other": 1})
        assert {t.name for t in result.backend_tools} == {"a", "b"}

    def test_scope_is_cached_per_page(self, scoped):
        first = scoped.get_available_tools({"page": {"name": "tasks"}})
        assert scoped.get_available_tools({"page": {"name": "tasks"}}) is first

    def test_warm_host_context_populates_caches(self, scoped):
        scoped.warm_host_context({"page": {"name": "tasks"}})
        assert "tasks" in scoped._available_tools_cache
        assert "tasks" in scoped._toolset_cache


class TestRequestDeclaredActions:
    """``host_context.actions`` become host tools for the turns that carry them."""

    def _context(self, *names, view="orders"):
        return {
            "view": {"name": view},
            "actions": [{"name": n, "description": f"Do {n}."} for n in names],
        }

    def test_actions_join_the_host_tools_and_toolsets(
        self, registry, backend_definition, dummy_handler
    ):
        registry.register_backend_tool(backend_definition, dummy_handler)
        available = registry.get_available_tools(self._context("open_order"))
        assert [t.name for t in available.host_tools] == ["open_order"]
        assert available.host_tools[0].category == ToolCategory.HOST
        assert available.page == "orders"
        toolsets = registry.build_toolset(self._context("open_order"))
        assert len(toolsets) == 2  # backend FunctionToolset + ExternalToolset for the action
        assert not registry.is_host_tool("open_order")  # per turn, not registered globally

    def test_without_actions_nothing_changes(self, registry, backend_definition, dummy_handler):
        registry.register_backend_tool(backend_definition, dummy_handler)
        available = registry.get_available_tools({"view": {"name": "orders"}})
        assert available.host_tools == []
        assert len(registry.build_toolset({"view": {"name": "orders"}})) == 1

    def test_contexts_with_actions_are_never_cached(self, registry):
        plain = registry.get_available_tools(self._context())
        with_action = registry.get_available_tools(self._context("open_order"))
        other_action = registry.get_available_tools(self._context("close_order"))
        assert plain.host_tools == []
        assert [t.name for t in with_action.host_tools] == ["open_order"]
        assert [t.name for t in other_action.host_tools] == ["close_order"]
        assert registry.get_available_tools(self._context()) is plain
        assert registry.get_available_tools(self._context("open_order")) is not with_action
        registry.build_toolset(self._context("open_order"))
        assert set(registry._toolset_cache) <= {"orders", None}
        assert set(registry._available_tools_cache) <= {"orders", None}

    def test_shadowing_a_registered_tool_is_ignored(self, backend_definition, dummy_handler):
        registry = ToolRegistry(
            ToolConfig(
                host_tools={"navigate": {"description": "Go", "parameters": {"type": "object"}}}
            )
        )
        registry.register_backend_tool(backend_definition, dummy_handler)
        registry.register_host_tools()
        available = registry.get_available_tools(self._context("get_time", "navigate", "fresh"))
        assert [t.name for t in available.host_tools] == ["navigate", "fresh"]
