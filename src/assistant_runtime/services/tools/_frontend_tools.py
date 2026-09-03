"""Frontend tool definitions for dashboard integration.

Frontend tools are executed by the host dashboard, not the backend.
They use Pydantic AI's ExternalToolset + DeferredToolRequests pattern:
1. LLM decides to call a frontend tool
2. Backend emits tool_call event with call_id
3. Frontend executes the tool (navigation, UI event, etc.)
4. Frontend sends a continuation request with the result
5. Backend resumes the agent with the tool result
"""

from __future__ import annotations

from typing import Any

from pydantic_ai.tools import ToolDefinition as PydanticToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition

# Frontend tool schemas — define what the LLM sees.
# These are always available regardless of page filtering.
FRONTEND_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "navigate": {
        "description": (
            "Navigate the dashboard to a specific page. "
            "Use this when the user asks to go to a page or when context requires "
            "viewing a different section."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "string",
                    "description": "Target page name (e.g., 'agents', 'tasks', 'meetings', 'repos', 'flows', 'home')",
                },
            },
            "required": ["page"],
        },
    },
    "ui_send_event": {
        "description": (
            "Send an event to the dashboard's XState machine. "
            "Use this to trigger UI actions like selecting items, "
            "filtering views, refreshing data, or changing state. "
            "Only use events listed in the 'Available UI actions' section "
            "of your context — those are the valid events for the current page and state. "
            "The data object fields MUST use the exact param names shown in the "
            "available actions list. Wrong field names cause silent failures "
            "(the event dispatches but the machine reads undefined values)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "description": (
                        "The XState event type to send (e.g., 'user.selectRoom', "
                        "'user.setFilter'). Must be one of the events listed in "
                        "'Available UI actions' in your context."
                    ),
                },
                "data": {
                    "type": "object",
                    "description": (
                        "Event payload — field names MUST match the param names "
                        "from 'Available UI actions' exactly. For example, if the "
                        'action lists \'id (string, required)\', use {"id": "value"}, '
                        'NOT {"roomId": "value"} or other variants.'
                    ),
                    "default": {},
                },
            },
            "required": ["event_type"],
        },
    },
}


def build_frontend_toolset() -> ExternalToolset | None:
    """Build a Pydantic AI ExternalToolset from frontend tool schemas.

    Returns None if no frontend tools are defined.
    """
    if not FRONTEND_TOOL_SCHEMAS:
        return None

    tool_defs = []
    for name, schema in FRONTEND_TOOL_SCHEMAS.items():
        tool_defs.append(
            PydanticToolDefinition(
                name=name,
                description=schema["description"],
                parameters_json_schema=schema["parameters"],
            )
        )

    return ExternalToolset(tool_defs)


def get_frontend_definitions() -> list[ToolDefinition]:
    """Return assistant-runtime ToolDefinition models for frontend tools.

    Used for ToolSet metadata (tool names, counts) — not for Pydantic AI toolset building.
    """
    definitions = []
    for name, schema in FRONTEND_TOOL_SCHEMAS.items():
        definitions.append(
            ToolDefinition(
                name=name,
                description=schema["description"],
                parameters_schema=schema["parameters"],
                category=ToolCategory.FRONTEND,
            )
        )
    return definitions
