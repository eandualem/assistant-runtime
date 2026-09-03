"""Screen inspection tools — on-demand visual inspection via ContextVar.

Images are captured per-request but NOT auto-sent to the LLM. The agent
uses ``look_at_screen`` to inspect the current screenshot when needed.
Screenshot data is stored in a request-scoped ContextVar (async-safe,
no DB dependency).
"""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from loguru import logger
from pydantic_ai.messages import BinaryContent

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

_current_screenshot: ContextVar[str | None] = ContextVar("_current_screenshot", default=None)
_SCREENSHOT_KEYS = (
    "screenshot",
    "image",
    "image_data_uri",
    "imageDataUri",
    "data_uri",
    "dataUri",
)


def set_current_screenshot(data_uri: str) -> None:
    """Store the current request's screenshot data URI."""
    _current_screenshot.set(data_uri)


def clear_current_screenshot() -> None:
    """Clear the stored screenshot (call in finally blocks)."""
    _current_screenshot.set(None)


def _find_screenshot_data_uri(value: Any) -> str | None:
    """Recursively search common payload shapes for an image data URI."""
    if isinstance(value, str):
        return value if value.startswith("data:image/") else None
    if isinstance(value, dict):
        for key in _SCREENSHOT_KEYS:
            if key in value:
                found = _find_screenshot_data_uri(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = _find_screenshot_data_uri(child)
            if found is not None:
                return found
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found = _find_screenshot_data_uri(item)
            if found is not None:
                return found
    return None


def extract_screenshot_data_uri(
    *,
    images: Sequence[str] | None = None,
    tool_result: Any | None = None,
) -> str | None:
    """Resolve the freshest screenshot from request images or continuation payloads."""
    if images:
        for image in images:
            if isinstance(image, str) and image:
                return image
    return _find_screenshot_data_uri(tool_result)


def strip_screenshot_from_tool_result(tool_result: Any) -> Any:
    """Remove screenshot data URIs from a tool_result dict.

    Frontend actions (navigate, click) may include a post-action screenshot
    in their response payload. We extract it into the ContextVar (via
    extract_screenshot_data_uri) but strip it before passing to the LLM
    so the base64 blob doesn't waste context tokens. The agent can still
    access the screenshot on-demand via look_at_screen.

    Returns a shallow copy with screenshot keys replaced by a placeholder.
    Non-dict values pass through unchanged.
    """
    if not isinstance(tool_result, dict):
        return tool_result

    cleaned = {}
    stripped = False
    for key, value in tool_result.items():
        if key in _SCREENSHOT_KEYS and isinstance(value, str) and value.startswith("data:image/"):
            cleaned[key] = "[screenshot captured — use look_at_screen to inspect]"
            stripped = True
        elif isinstance(value, dict):
            cleaned[key] = strip_screenshot_from_tool_result(value)
        else:
            cleaned[key] = value

    if stripped:
        logger.debug("Stripped screenshot from tool_result for LLM context efficiency")
    return cleaned


async def _look_at_screen() -> BinaryContent | dict[str, str]:
    """Inspect the current dashboard screenshot.

    Returns the screenshot as BinaryContent (rendered as an image for the
    model) or an error dict if no screenshot is available.
    """
    data_uri = _current_screenshot.get()
    if data_uri is None:
        return {
            "error": (
                "No screenshot available for this request. "
                "The frontend must include the screenshot as a data URI in "
                "images[], or as a top-level 'screenshot' field in the request. "
                "After UI navigation, the next message must include a fresh capture."
            ),
            "error_code": "NO_SCREENSHOT",
        }
    try:
        return BinaryContent.from_data_uri(data_uri)
    except Exception as e:
        logger.warning("Failed to parse screenshot data URI", error=str(e))
        return {
            "error": f"Invalid screenshot data: {e}",
            "error_code": "INVALID_SCREENSHOT",
        }


def register_screen_tools(registry: ToolRegistry) -> None:
    """Register screen inspection tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="look_at_screen",
            description=(
                "Look at the user's current screen. Returns the latest dashboard "
                "screenshot. Use this after navigation or UI actions to visually "
                "confirm the result — layout, data displayed, error states, etc. "
                "Frontend actions (navigate, ui_send_event) include a post-action "
                "screenshot automatically. Call this tool to inspect it."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        _look_at_screen,
    )
