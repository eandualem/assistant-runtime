"""``look_at_screen`` — on-demand inspection of the screenshot the host attached.

Screenshots are never sent to the model automatically; the host attaches
one to the request (an attachment with ``purpose: "screenshot"``, or the
legacy ``images[]`` / top-level ``screenshot`` field), the turn pipeline
binds it to the request context, and the model calls this tool when it
wants to see it. Reference attachments, by contrast, go straight into the
message as native content.
"""

from __future__ import annotations

from loguru import logger
from pydantic_ai.messages import BinaryContent

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools._request_context import get_current_screenshot
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


async def look_at_screen() -> BinaryContent | dict[str, str]:
    """The current screenshot as image content, or an error dict when there is none."""
    data_uri = get_current_screenshot()
    if data_uri is None:
        return {
            "error": (
                "No screenshot available for this request. "
                "The host must attach the screenshot as a data URI (an attachment "
                "with purpose 'screenshot', or the legacy images[] field). "
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
                "Look at the user's current screen. Returns the latest screenshot "
                "the host attached. Use this after navigation or UI actions to visually "
                "confirm the result — layout, data displayed, error states, etc. "
                "Host actions (navigate, ui_send_event) include a post-action "
                "screenshot automatically. Call this tool to inspect it."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        look_at_screen,
    )
