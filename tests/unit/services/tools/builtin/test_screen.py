"""Tests for screen inspection tools — ContextVar management and handler behavior."""

from __future__ import annotations

import base64

import pytest
from pydantic_ai.messages import BinaryContent

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools._request_context import (
    assistant_request_context,
    get_current_screenshot,
)
from assistant_runtime.services.tools.builtin.screen import look_at_screen, register_screen_tools
from assistant_runtime.services.tools.config import ToolConfig

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


@pytest.fixture
def registry():
    return ToolRegistry(ToolConfig())


class TestRequestContextScreenshot:
    """The screenshot is bound for the duration of one request context."""

    def test_default_is_none(self):
        assert get_current_screenshot() is None

    def test_bound_inside_the_context_only(self):
        with assistant_request_context("sess-1", screenshot="data:image/png;base64,abc"):
            assert get_current_screenshot() == "data:image/png;base64,abc"
        assert get_current_screenshot() is None

    def test_context_without_screenshot_hides_an_outer_one(self):
        with assistant_request_context("sess-1", screenshot="data:first"):
            with assistant_request_context("sess-1"):
                assert get_current_screenshot() is None
            assert get_current_screenshot() == "data:first"


# ---------------------------------------------------------------------------
# TestLookAtScreenNoScreenshot
# ---------------------------------------------------------------------------


class TestLookAtScreenNoScreenshot:
    """look_at_screen returns error dict when no screenshot is set."""

    async def test_returns_dict_when_no_screenshot(self):
        result = await look_at_screen()
        assert isinstance(result, dict)

    async def test_error_message(self):
        result = await look_at_screen()
        assert "No screenshot available" in result["error"]
        assert "screenshot" in result["error"]

    async def test_error_code(self):
        result = await look_at_screen()
        assert result["error_code"] == "NO_SCREENSHOT"

    async def test_returns_exactly_two_keys(self):
        result = await look_at_screen()
        assert set(result.keys()) == {"error", "error_code"}


# ---------------------------------------------------------------------------
# TestLookAtScreenValidDataUri
# ---------------------------------------------------------------------------


class TestLookAtScreenValidDataUri:
    """look_at_screen returns BinaryContent when a valid data URI is set."""

    async def test_returns_binary_content(self):
        with assistant_request_context("sess-1", screenshot=VALID_DATA_URI):
            result = await look_at_screen()
        assert isinstance(result, BinaryContent)

    async def test_not_a_dict(self):
        with assistant_request_context("sess-1", screenshot=VALID_DATA_URI):
            result = await look_at_screen()
        assert not isinstance(result, dict)


# ---------------------------------------------------------------------------
# TestLookAtScreenInvalidDataUri
# ---------------------------------------------------------------------------


class TestLookAtScreenInvalidDataUri:
    """look_at_screen returns error dict for an invalid data URI."""

    async def test_returns_dict(self):
        with assistant_request_context("sess-1", screenshot="data:invalid"):
            result = await look_at_screen()
        assert isinstance(result, dict)

    async def test_has_error_key(self):
        with assistant_request_context("sess-1", screenshot="data:invalid"):
            result = await look_at_screen()
        assert "error" in result

    async def test_has_error_code_key(self):
        with assistant_request_context("sess-1", screenshot="data:invalid"):
            result = await look_at_screen()
        assert "error_code" in result

    async def test_error_code_is_invalid_screenshot(self):
        with assistant_request_context("sess-1", screenshot="data:invalid"):
            result = await look_at_screen()
        assert result["error_code"] == "INVALID_SCREENSHOT"


class TestRegisterScreenTools:
    """register_screen_tools registers exactly one backend tool."""

    def test_registers_one_tool(self, registry):
        register_screen_tools(registry)
        assert registry.backend_tool_count() == 1

    def testlook_at_screen_in_tool_names(self, registry):
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
