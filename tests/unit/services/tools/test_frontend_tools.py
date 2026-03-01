"""Tests for frontend tool definitions module."""

from pydantic_ai.toolsets import ExternalToolset

from lovely_assistant.services.tools._frontend_tools import (
    FRONTEND_TOOL_SCHEMAS,
    build_frontend_toolset,
    get_frontend_definitions,
)
from lovely_assistant.services.tools.models import ToolCategory

# ---------------------------------------------------------------------------
# TestFrontendToolSchemas
# ---------------------------------------------------------------------------


class TestFrontendToolSchemas:
    def test_has_exactly_two_entries(self):
        assert len(FRONTEND_TOOL_SCHEMAS) == 2

    def test_expected_tool_names_present(self):
        assert "navigate" in FRONTEND_TOOL_SCHEMAS
        assert "ui_send_event" in FRONTEND_TOOL_SCHEMAS

    def test_each_schema_has_description(self):
        for name, schema in FRONTEND_TOOL_SCHEMAS.items():
            assert "description" in schema, f"Schema for '{name}' missing 'description'"
            assert schema["description"], f"Schema for '{name}' has empty description"

    def test_each_schema_has_parameters(self):
        for name, schema in FRONTEND_TOOL_SCHEMAS.items():
            assert "parameters" in schema, f"Schema for '{name}' missing 'parameters'"
            assert isinstance(schema["parameters"], dict), (
                f"Schema for '{name}' has non-dict parameters"
            )


# ---------------------------------------------------------------------------
# TestBuildFrontendToolset
# ---------------------------------------------------------------------------


class TestBuildFrontendToolset:
    def test_returns_external_toolset(self):
        result = build_frontend_toolset()
        assert isinstance(result, ExternalToolset)

    def test_not_none(self):
        result = build_frontend_toolset()
        assert result is not None


# ---------------------------------------------------------------------------
# TestGetFrontendDefinitions
# ---------------------------------------------------------------------------


class TestGetFrontendDefinitions:
    def test_returns_two_definitions(self):
        definitions = get_frontend_definitions()
        assert len(definitions) == 2

    def test_all_have_frontend_category(self):
        definitions = get_frontend_definitions()
        for defn in definitions:
            assert defn.category == ToolCategory.FRONTEND

    def test_all_have_non_empty_description(self):
        definitions = get_frontend_definitions()
        for defn in definitions:
            assert defn.description, f"Definition '{defn.name}' has empty description"

    def test_all_have_non_empty_parameters_schema(self):
        definitions = get_frontend_definitions()
        for defn in definitions:
            assert isinstance(defn.parameters_schema, dict), (
                f"Definition '{defn.name}' has non-dict parameters_schema"
            )

    def test_names_match_frontend_tool_schemas_keys(self):
        definitions = get_frontend_definitions()
        definition_names = {defn.name for defn in definitions}
        schema_names = set(FRONTEND_TOOL_SCHEMAS.keys())
        assert definition_names == schema_names

    def test_navigate_definition_present(self):
        definitions = get_frontend_definitions()
        names = [defn.name for defn in definitions]
        assert "navigate" in names

    def test_ui_send_event_definition_present(self):
        definitions = get_frontend_definitions()
        names = [defn.name for defn in definitions]
        assert "ui_send_event" in names
