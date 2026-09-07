"""Tests for the screenshot payload helpers on the request model."""

from __future__ import annotations

from assistant_runtime.app.assistant.models import (
    extract_screenshot_data_uri,
    strip_screenshot_from_tool_result,
)

VALID_DATA_URI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2Q=="


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


class TestStripScreenshotFromToolResult:
    """strip_screenshot_from_tool_result removes data URIs from tool_result."""

    def test_strips_top_level_screenshot_key(self):
        result = strip_screenshot_from_tool_result(
            {
                "success": True,
                "screenshot": VALID_DATA_URI,
            }
        )
        assert result["success"] is True
        assert "look_at_screen" in result["screenshot"]
        assert "data:image/" not in result["screenshot"]

    def test_strips_nested_screenshot(self):
        result = strip_screenshot_from_tool_result(
            {
                "success": True,
                "data": {"screenshot": VALID_DATA_URI},
            }
        )
        assert result["success"] is True
        assert "look_at_screen" in result["data"]["screenshot"]

    def test_strips_screenshots_inside_lists(self):
        result = strip_screenshot_from_tool_result(
            {"steps": [{"screenshot": VALID_DATA_URI}, {"status": "ok"}]}
        )
        assert "look_at_screen" in result["steps"][0]["screenshot"]
        assert result["steps"][1] == {"status": "ok"}

    def test_strips_data_uris_under_any_key(self):
        result = strip_screenshot_from_tool_result({"result": VALID_DATA_URI, "ok": True})
        assert "look_at_screen" in result["result"]
        assert result["ok"] is True

    def test_preserves_non_screenshot_keys(self):
        result = strip_screenshot_from_tool_result(
            {
                "success": True,
                "page": "sessions",
                "message": "Navigated",
            }
        )
        assert result == {"success": True, "page": "sessions", "message": "Navigated"}

    def test_non_data_uri_not_stripped(self):
        result = strip_screenshot_from_tool_result(
            {
                "screenshot": "just-a-filename.png",
            }
        )
        assert result["screenshot"] == "just-a-filename.png"

    def test_non_dict_passthrough(self):
        assert strip_screenshot_from_tool_result("hello") == "hello"
        assert "look_at_screen" in strip_screenshot_from_tool_result(VALID_DATA_URI)
        assert strip_screenshot_from_tool_result(42) == 42
        assert strip_screenshot_from_tool_result(None) is None

    def test_empty_dict(self):
        assert strip_screenshot_from_tool_result({}) == {}
