"""Tests for tool service data models — ToolCategory, ToolDefinition, ToolSet."""

import pytest
from pydantic import ValidationError

from assistant_runtime.services.tools.models import (
    ToolCategory,
    ToolDefinition,
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
