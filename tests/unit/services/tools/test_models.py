"""Tests for tool service data models — ToolCategory, ToolDefinition, ToolSet, ToolResult."""

import pytest
from pydantic import ValidationError

from lovely_assistant.services.tools.models import (
    ToolCategory,
    ToolDefinition,
    ToolResult,
    ToolSet,
)


class TestToolCategory:
    def test_values(self):
        assert ToolCategory.BACKEND == "backend"

    def test_is_str(self):
        assert isinstance(ToolCategory.BACKEND, str)


class TestToolDefinition:
    def test_creation(self):
        defn = ToolDefinition(
            name="get_time",
            description="Returns current UTC time",
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        )
        assert defn.name == "get_time"
        assert defn.description == "Returns current UTC time"
        assert defn.parameters_schema == {"type": "object", "properties": {}}
        assert defn.category == ToolCategory.BACKEND

    def test_frozen(self):
        defn = ToolDefinition(
            name="get_time",
            description="Returns current UTC time",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        with pytest.raises(ValidationError):
            defn.name = "changed"

    def test_optional_timeout(self):
        defn = ToolDefinition(
            name="test",
            description="A test tool",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        assert defn.timeout is None

    def test_with_timeout(self):
        defn = ToolDefinition(
            name="slow_tool",
            description="A slow tool",
            parameters_schema={},
            category=ToolCategory.BACKEND,
            timeout=60.0,
        )
        assert defn.timeout == 60.0


class TestToolSet:
    def test_empty(self):
        ts = ToolSet()
        assert ts.total_count == 0
        assert ts.tool_names == []

    def test_with_tools(self):
        backend = ToolDefinition(
            name="get_time",
            description="Time",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        ts = ToolSet(backend_tools=[backend])
        assert ts.total_count == 1
        assert "get_time" in ts.tool_names

    def test_tool_names_order(self):
        """Tool names appear in insertion order."""
        b1 = ToolDefinition(
            name="backend_a",
            description="B-A",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        b2 = ToolDefinition(
            name="backend_b",
            description="B-B",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        ts = ToolSet(backend_tools=[b1, b2])
        assert ts.tool_names == ["backend_a", "backend_b"]


class TestToolResult:
    def test_creation(self):
        result = ToolResult(
            tool_name="get_time",
            request_id="abc-123",
            content="2026-02-18T12:00:00Z",
        )
        assert result.tool_name == "get_time"
        assert result.request_id == "abc-123"
        assert result.content == "2026-02-18T12:00:00Z"

    def test_default_is_error(self):
        result = ToolResult(
            tool_name="get_time",
            request_id="abc-123",
            content="ok",
        )
        assert result.is_error is False

    def test_error_result(self):
        result = ToolResult(
            tool_name="get_time",
            request_id="abc-123",
            content="Connection refused",
            is_error=True,
        )
        assert result.is_error is True
        assert result.content == "Connection refused"


class TestToolCategoryFrontend:
    def test_frontend_category_exists(self):
        assert ToolCategory.FRONTEND == "frontend"

    def test_frontend_tool_definition(self):
        defn = ToolDefinition(
            name="navigate",
            description="Navigate to a dashboard page.",
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.FRONTEND,
        )
        assert defn.category == ToolCategory.FRONTEND
        assert defn.name == "navigate"


class TestToolSetFrontendTools:
    def test_total_count_includes_frontend(self):
        backend_a = ToolDefinition(
            name="tool_a",
            description="Backend tool A",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        backend_b = ToolDefinition(
            name="tool_b",
            description="Backend tool B",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        frontend_c = ToolDefinition(
            name="tool_c",
            description="Frontend tool C",
            parameters_schema={},
            category=ToolCategory.FRONTEND,
        )
        ts = ToolSet(backend_tools=[backend_a, backend_b], frontend_tools=[frontend_c])
        assert ts.total_count == 3

    def test_tool_names_includes_frontend(self):
        backend = ToolDefinition(
            name="a",
            description="Backend tool",
            parameters_schema={},
            category=ToolCategory.BACKEND,
        )
        frontend = ToolDefinition(
            name="b",
            description="Frontend tool",
            parameters_schema={},
            category=ToolCategory.FRONTEND,
        )
        ts = ToolSet(backend_tools=[backend], frontend_tools=[frontend])
        assert ts.tool_names == ["a", "b"]

    def test_empty_frontend_default(self):
        ts = ToolSet()
        assert ts.frontend_tools == []
