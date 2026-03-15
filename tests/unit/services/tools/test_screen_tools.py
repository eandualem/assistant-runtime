"""Tests for screen inspection tools — ContextVar management and handler behavior."""

from __future__ import annotations

import base64

import pytest
from pydantic_ai.messages import BinaryContent

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._screen_tools import (
    _current_screenshot,
    _look_at_screen,
    clear_current_screenshot,
    extract_screenshot_data_uri,
    register_screen_tools,
    set_current_screenshot,
    strip_screenshot_from_tool_result,
)
from lovely_assistant.services.tools.config import ToolConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_TINY_JPEG = base64.b64encode(
    bytes(
        [
            0xFF,
            0xD8,
            0xFF,
            0xE0,
            0x00,
            0x10,
            0x4A,
            0x46,
            0x49,
            0x46,
            0x00,
            0x01,
            0x01,
            0x00,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x00,
            0xFF,
            0xD9,
        ]
    )
).decode()

VALID_DATA_URI = f"data:image/jpeg;base64,{_TINY_JPEG}"


@pytest.fixture(autouse=True)
def _clear_screenshot():
    """Ensure the ContextVar is cleared before and after every test."""
    clear_current_screenshot()
    yield
    clear_current_screenshot()


@pytest.fixture
def registry():
    return ToolRegistry(ToolConfig())


# ---------------------------------------------------------------------------
# TestContextVarSetClear
# ---------------------------------------------------------------------------


class TestContextVarSetClear:
    """ContextVar set/clear behavior."""

    def test_set_stores_value(self):
        set_current_screenshot("data:image/png;base64,abc")
        assert _current_screenshot.get() == "data:image/png;base64,abc"

    def test_clear_resets_to_none(self):
        set_current_screenshot("data:image/png;base64,abc")
        clear_current_screenshot()
        assert _current_screenshot.get() is None

    def test_default_is_none(self):
        assert _current_screenshot.get() is None

    def test_set_overwrites_previous(self):
        set_current_screenshot("data:first")
        set_current_screenshot("data:second")
        assert _current_screenshot.get() == "data:second"


# ---------------------------------------------------------------------------
# TestLookAtScreenNoScreenshot
# ---------------------------------------------------------------------------


class TestLookAtScreenNoScreenshot:
    """_look_at_screen returns error dict when no screenshot is set."""

    async def test_returns_dict_when_no_screenshot(self):
        result = await _look_at_screen()
        assert isinstance(result, dict)

    async def test_error_message(self):
        result = await _look_at_screen()
        assert "No screenshot available" in result["error"]
        assert "screenshot" in result["error"]

    async def test_error_code(self):
        result = await _look_at_screen()
        assert result["error_code"] == "NO_SCREENSHOT"

    async def test_returns_exactly_two_keys(self):
        result = await _look_at_screen()
        assert set(result.keys()) == {"error", "error_code"}


# ---------------------------------------------------------------------------
# TestLookAtScreenValidDataUri
# ---------------------------------------------------------------------------


class TestLookAtScreenValidDataUri:
    """_look_at_screen returns BinaryContent when a valid data URI is set."""

    async def test_returns_binary_content(self):
        set_current_screenshot(VALID_DATA_URI)
        result = await _look_at_screen()
        assert isinstance(result, BinaryContent)

    async def test_not_a_dict(self):
        set_current_screenshot(VALID_DATA_URI)
        result = await _look_at_screen()
        assert not isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestLookAtScreenInvalidDataUri
# ---------------------------------------------------------------------------


class TestLookAtScreenInvalidDataUri:
    """_look_at_screen returns error dict for an invalid data URI."""

    async def test_returns_dict(self):
        set_current_screenshot("data:invalid")
        result = await _look_at_screen()
        assert isinstance(result, dict)

    async def test_has_error_key(self):
        set_current_screenshot("data:invalid")
        result = await _look_at_screen()
        assert "error" in result

    async def test_has_error_code_key(self):
        set_current_screenshot("data:invalid")
        result = await _look_at_screen()
        assert "error_code" in result

    async def test_error_code_is_invalid_screenshot(self):
        set_current_screenshot("data:invalid")
        result = await _look_at_screen()
        assert result["error_code"] == "INVALID_SCREENSHOT"


class TestExtractScreenshotDataUri:
    """Continuation payloads may carry screenshots outside top-level request.images."""

    def test_prefers_request_images(self):
        result = extract_screenshot_data_uri(
            images=["data:image/png;base64,top-level"],
            tool_result={"screenshot": "data:image/png;base64,nested"},
        )
        assert result == "data:image/png;base64,top-level"

    def test_extracts_direct_screenshot_field(self):
        result = extract_screenshot_data_uri(
            tool_result={"screenshot": "data:image/png;base64,from-result"}
        )
        assert result == "data:image/png;base64,from-result"

    def test_extracts_nested_image_data_uri_field(self):
        result = extract_screenshot_data_uri(
            tool_result={"result": {"imageDataUri": "data:image/png;base64,nested"}}
        )
        assert result == "data:image/png;base64,nested"

    def test_returns_none_when_no_image_data_uri_present(self):
        result = extract_screenshot_data_uri(tool_result={"status": "ok"})
        assert result is None


# ---------------------------------------------------------------------------
# TestRegisterScreenTools
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# strip_screenshot_from_tool_result
# ---------------------------------------------------------------------------


class TestStripScreenshotFromToolResult:
    """strip_screenshot_from_tool_result removes data URIs from tool_result."""

    def test_strips_top_level_screenshot_key(self):
        result = strip_screenshot_from_tool_result({
            "success": True,
            "screenshot": VALID_DATA_URI,
        })
        assert result["success"] is True
        assert "look_at_screen" in result["screenshot"]
        assert "data:image/" not in result["screenshot"]

    def test_strips_nested_screenshot(self):
        result = strip_screenshot_from_tool_result({
            "success": True,
            "data": {"screenshot": VALID_DATA_URI},
        })
        assert result["success"] is True
        assert "look_at_screen" in result["data"]["screenshot"]

    def test_preserves_non_screenshot_keys(self):
        result = strip_screenshot_from_tool_result({
            "success": True,
            "page": "sessions",
            "message": "Navigated",
        })
        assert result == {"success": True, "page": "sessions", "message": "Navigated"}

    def test_non_data_uri_not_stripped(self):
        result = strip_screenshot_from_tool_result({
            "screenshot": "just-a-filename.png",
        })
        assert result["screenshot"] == "just-a-filename.png"

    def test_non_dict_passthrough(self):
        assert strip_screenshot_from_tool_result("hello") == "hello"
        assert strip_screenshot_from_tool_result(42) == 42
        assert strip_screenshot_from_tool_result(None) is None

    def test_empty_dict(self):
        assert strip_screenshot_from_tool_result({}) == {}


class TestRegisterScreenTools:
    """register_screen_tools registers exactly one backend tool."""

    def test_registers_one_tool(self, registry):
        register_screen_tools(registry)
        assert registry.backend_tool_count() == 1

    def test_look_at_screen_in_tool_names(self, registry):
        register_screen_tools(registry)
        assert "look_at_screen" in registry.get_tool_names()

    def test_has_handler(self, registry):
        register_screen_tools(registry)
        assert callable(registry._backend_handlers["look_at_screen"])

    def test_has_definition(self, registry):
        register_screen_tools(registry)
        defn = registry._backend_definitions["look_at_screen"]
        assert defn.name == "look_at_screen"
        assert defn.description
