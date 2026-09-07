"""Repositories: the code repositories agents are set up for.

A provider implements ``RepositoriesProvider``; each method returns the tool's result
dict (``{"success": False, "error": ...}`` on failure).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


class RepositoriesProvider(Protocol):
    """What a provider of the repositories capability implements."""

    async def onboard_repo(self, org: str, url: str) -> dict[str, Any]: ...

    async def check_repo_status(self, org: str, repo: str) -> dict[str, Any]: ...

    async def list_repos(self) -> dict[str, Any]: ...


def register_repositories_tools(registry: ToolRegistry, provider: RepositoriesProvider) -> None:
    """Register all repo management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="onboard_repo",
            description=(
                "Onboard a new repository into the workspace. Clones the repo, "
                "sets up CLAUDE.md, .claude/ directory, settings, and registers it "
                "in the agent backbone. Requires the org name "
                "and the Git URL."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "org": {
                        "type": "string",
                        "description": "Organization the repo belongs to",
                    },
                    "url": {
                        "type": "string",
                        "description": "Git URL of the repository to onboard",
                    },
                },
                "required": ["org", "url"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.onboard_repo,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="check_repo_status",
            description=(
                "Check the onboarding and configuration status of a specific "
                "repository. Returns setup steps completed, missing config, "
                "and current state."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "org": {
                        "type": "string",
                        "description": "Organization the repo belongs to",
                    },
                    "repo": {
                        "type": "string",
                        "description": "Repository name",
                    },
                },
                "required": ["org", "repo"],
            },
            category=ToolCategory.BACKEND,
        ),
        provider.check_repo_status,
    )

    registry.register_backend_tool(
        ToolDefinition(
            name="list_repos",
            description=(
                "List all managed repositories across all organizations. "
                "Returns repo names, orgs, and basic status for each."
            ),
            parameters_schema={"type": "object", "properties": {}},
            category=ToolCategory.BACKEND,
        ),
        provider.list_repos,
    )

    logger.info("Registered repo management tools", count=3)
