"""Host tool schemas: config-driven, validated, built into an ExternalToolset."""

import json

import pytest
from pydantic_ai.toolsets import ExternalToolset

from assistant_runtime.services.tools._host_tools import (
    build_host_toolset,
    get_host_definitions,
    load_host_tool_schemas,
)
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolValidationError
from assistant_runtime.services.tools.models import ToolCategory

NAVIGATE = {
    "description": "Navigate the host to a page.",
    "parameters": {"type": "object", "properties": {"page": {"type": "string"}}},
}


class TestLoadHostToolSchemas:
    def test_empty_by_default(self):
        assert load_host_tool_schemas(ToolConfig()) == {}

    def test_from_config_dict(self):
        assert load_host_tool_schemas(ToolConfig(host_tools={"navigate": NAVIGATE})) == {
            "navigate": NAVIGATE
        }

    def test_from_file(self, tmp_path):
        path = tmp_path / "host_tools.json"
        path.write_text(json.dumps({"select": NAVIGATE}))
        schemas = load_host_tool_schemas(ToolConfig(host_tools_path=str(path)))
        assert list(schemas) == ["select"]

    def test_file_overrides_dict(self, tmp_path):
        path = tmp_path / "host_tools.json"
        override = {**NAVIGATE, "description": "From file"}
        path.write_text(json.dumps({"navigate": override}))
        config = ToolConfig(host_tools={"navigate": NAVIGATE}, host_tools_path=str(path))
        assert load_host_tool_schemas(config)["navigate"]["description"] == "From file"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ToolValidationError):
            load_host_tool_schemas(ToolConfig(host_tools_path=str(tmp_path / "nope.json")))

    def test_file_must_be_object(self, tmp_path):
        path = tmp_path / "host_tools.json"
        path.write_text("[]")
        with pytest.raises(ToolValidationError):
            load_host_tool_schemas(ToolConfig(host_tools_path=str(path)))

    def test_description_required(self):
        with pytest.raises(ToolValidationError):
            load_host_tool_schemas(ToolConfig(host_tools={"x": {"parameters": {}}}))

    @pytest.mark.parametrize(
        "name", ["", "has space", "dot.name", "x" * 65, "ünïcode", "navigate\n"]
    )
    def test_invalid_names_rejected(self, name):
        with pytest.raises(ToolValidationError):
            load_host_tool_schemas(ToolConfig(host_tools={name: NAVIGATE}))

    def test_parameters_must_be_object(self):
        with pytest.raises(ToolValidationError):
            load_host_tool_schemas(ToolConfig(host_tools={"x": {"description": "d"}}))


class TestBuildHostToolset:
    def test_none_when_empty(self):
        assert build_host_toolset({}) is None

    def test_external_toolset(self):
        assert isinstance(build_host_toolset({"navigate": NAVIGATE}), ExternalToolset)


class TestGetHostDefinitions:
    def test_empty(self):
        assert get_host_definitions({}) == []

    def test_definitions_are_frontend_category(self):
        defs = get_host_definitions({"navigate": NAVIGATE})
        assert [d.name for d in defs] == ["navigate"]
        assert defs[0].category == ToolCategory.FRONTEND
        assert defs[0].description == NAVIGATE["description"]
        assert defs[0].parameters_schema == NAVIGATE["parameters"]
