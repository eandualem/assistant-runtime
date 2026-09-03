"""Tests for ToolConfig validation, defaults, and bounds."""

import pytest
from pydantic import ValidationError

from assistant_runtime.services.tools.config import ToolConfig


class TestToolConfigDefaults:
    def test_defaults(self):
        config = ToolConfig()
        assert config.max_tools_per_request == 40
        assert config.tool_timeout_seconds == 30.0

    def test_custom_values(self):
        config = ToolConfig(
            max_tools_per_request=50,
            tool_timeout_seconds=120.0,
        )
        assert config.max_tools_per_request == 50
        assert config.tool_timeout_seconds == 120.0


class TestToolConfigValidation:
    def test_frozen(self):
        config = ToolConfig()
        with pytest.raises(ValidationError):
            config.max_tools_per_request = 10

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ToolConfig(nonexistent_field="value")

    def test_max_tools_below_min_rejected(self):
        with pytest.raises(ValidationError):
            ToolConfig(max_tools_per_request=0)

    def test_max_tools_above_max_rejected(self):
        with pytest.raises(ValidationError):
            ToolConfig(max_tools_per_request=101)

    def test_max_tools_boundary_min(self):
        config = ToolConfig(max_tools_per_request=1)
        assert config.max_tools_per_request == 1

    def test_max_tools_boundary_max(self):
        config = ToolConfig(max_tools_per_request=100)
        assert config.max_tools_per_request == 100

    def test_timeout_below_min_rejected(self):
        with pytest.raises(ValidationError):
            ToolConfig(tool_timeout_seconds=0.5)

    def test_timeout_above_max_rejected(self):
        with pytest.raises(ValidationError):
            ToolConfig(tool_timeout_seconds=301.0)

    def test_timeout_boundary_min(self):
        config = ToolConfig(tool_timeout_seconds=1.0)
        assert config.tool_timeout_seconds == 1.0

    def test_timeout_boundary_max(self):
        config = ToolConfig(tool_timeout_seconds=300.0)
        assert config.tool_timeout_seconds == 300.0
