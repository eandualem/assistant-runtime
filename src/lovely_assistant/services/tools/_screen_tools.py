"""Screen inspection tools — on-demand visual inspection via ContextVar.

Images are captured per-request but NOT auto-sent to the LLM. The agent
uses ``look_at_screen`` to inspect the current screenshot when needed.
Screenshot data is stored in a request-scoped ContextVar (async-safe,
no DB dependency).
"""

from __future__ import annotations

from contextvars import ContextVar

from loguru import logger
from pydantic_ai.messages import BinaryContent

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

_current_screenshot: ContextVar[str | None] = ContextVar("_current_screenshot", default=None)


def set_current_screenshot(data_uri: str) -> None:
    """Store the current request's screenshot data URI."""
    _current_screenshot.set(data_uri)


def clear_current_screenshot() -> None:
    """Clear the stored screenshot (call in finally blocks)."""
    _current_screenshot.set(None)


async def _look_at_screen() -> BinaryContent | dict[str, str]:
    """Inspect the current dashboard screenshot.

    Returns the screenshot as BinaryContent (rendered as an image for the
    model) or an error dict if no screenshot is available.
    """
    data_uri = _current_screenshot.get()
    if data_uri is None:
        return {
            "error": "No screenshot available for this request.",
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
                "screenshot. Use this when you need to visually inspect what the user "
                "sees — layout, data displayed, error states, etc. The screenshot is "
                "captured automatically with each message but only sent to you when "
                "you call this tool."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        _look_at_screen,
    )
