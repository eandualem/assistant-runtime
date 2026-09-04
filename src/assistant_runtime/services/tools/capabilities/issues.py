"""Issues: the issue tracker the assistant files and follows work in.

Create, search, read, comment on and close issues.

A provider implements ``IssuesProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class IssuesProvider(Protocol):
    """What a provider of the issues capability implements."""

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
        priority: str = "",
    ) -> dict[str, Any]: ...

    async def search_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
        text: str = "",
        limit: int = 20,
    ) -> dict[str, Any]: ...

    async def get_issue_details(self, issue_number: int) -> dict[str, Any]: ...

    async def comment_on_issue(self, issue_number: int, body: str) -> dict[str, Any]: ...

    async def close_issue(self, issue_number: int, comment: str = "") -> dict[str, Any]: ...


def register_issues_tools(registry: ToolRegistry, provider: IssuesProvider) -> None:
    """Register all GitHub issue management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="create_issue",
            description=(
                "Create a new issue in the orchestration repository. "
                "Labels must include at least one 'from:' and one 'for:' label "
                "per system convention. Optionally set priority to 'blocking' or 'non-blocking'."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Issue title, e.g. '[task] Brief description'",
                    },
                    "body": {
                        "type": "string",
                        "description": "Issue body in markdown with ## Context, ## Request, ## References sections",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels including from: and for: labels, plus a type label (task, bug, spec-gap, question, optimization)",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["blocking", "non-blocking", ""],
                        "description": "Optional priority label",
                        "default": "",
                    },
                },
                "required": ["title", "body", "labels"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.create_issue,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="search_issues",
            description=(
                "Search issues in the orchestration repository. "
                "Can filter by state, labels, and text query. "
                "Returns issue number, title, state, labels, and creation date."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Issue state filter",
                        "default": "open",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by labels (e.g. ['for:coding-agent', 'task'])",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text search query",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 20,
                    },
                },
            },
            category=ToolCategory.BACKEND,
        ),
        provider.search_issues,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="get_issue_details",
            description=(
                "Get full details of a specific issue including body, labels, "
                "and all comments with authors and timestamps."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue number to retrieve",
                    },
                },
                "required": ["issue_number"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.get_issue_details,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="comment_on_issue",
            description=(
                "Add a comment to an existing issue. Used for acknowledgments, "
                "status updates, and closing remarks."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue number to comment on",
                    },
                    "body": {
                        "type": "string",
                        "description": "Comment text in markdown",
                    },
                },
                "required": ["issue_number", "body"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.comment_on_issue,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="close_issue",
            description=(
                "Close an issue with state_reason 'completed'. "
                "Optionally adds a closing comment before closing."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "issue_number": {
                        "type": "integer",
                        "description": "The issue number to close",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Optional closing comment to add before closing",
                        "default": "",
                    },
                },
                "required": ["issue_number"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.close_issue,
    )

    logger.info("Registered GitHub issue management tools", count=5)
