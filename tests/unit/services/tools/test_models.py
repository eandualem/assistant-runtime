"""Tests for tool service data models — ToolCategory, ToolDefinition, ToolSet, DeferredToolRequest, ToolResult."""

import re

import pytest
from pydantic import ValidationError

from lovely_assistant.services.tools.models import (
    DeferredToolRequest,
    ToolCategory,
    ToolDefinition,
    ToolResult,
    ToolSet,
)


class TestToolCategory:
    def test_values(self):
        assert ToolCategory.BACKEND == "backend"
        assert ToolCategory.FRONTEND == "frontend"

    def test_is_str(self):
        assert isinstance(ToolCategory.BACKEND, str)
        assert isinstance(ToolCategory.FRONTEND, str)


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
            category=ToolCategory.FRONTEND,
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
        frontend = ToolDefinition(
            name="navigate",
            description="Notify",
            parameters_schema={},
            category=ToolCategory.FRONTEND,
        )
        ts = ToolSet(backend_tools=[backend], frontend_tools=[frontend])
        assert ts.total_count == 2
        assert "get_time" in ts.tool_names
        assert "navigate" in ts.tool_names

    def test_tool_names_order(self):
        """Backend tool names appear before frontend tool names."""
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
        f1 = ToolDefinition(
            name="frontend_x",
            description="F-X",
            parameters_schema={},
            category=ToolCategory.FRONTEND,
        )
        ts = ToolSet(backend_tools=[b1, b2], frontend_tools=[f1])
        assert ts.tool_names == ["backend_a", "backend_b", "frontend_x"]


class TestDeferredToolRequest:
    def test_creation(self):
        req = DeferredToolRequest(
            tool_name="ui_navigate",
            arguments={"path": "/dashboard"},
        )
        assert req.tool_name == "ui_navigate"
        assert req.arguments == {"path": "/dashboard"}

    def test_auto_request_id(self):
        req = DeferredToolRequest(tool_name="navigate")
        assert req.request_id is not None
        # UUID v4 format: 8-4-4-4-12 hex characters
        uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
        assert uuid_pattern.match(req.request_id)

    def test_unique_ids(self):
        req1 = DeferredToolRequest(tool_name="tool_a")
        req2 = DeferredToolRequest(tool_name="tool_b")
        assert req1.request_id != req2.request_id

    def test_default_arguments(self):
        req = DeferredToolRequest(tool_name="navigate")
        assert req.arguments == {}


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
