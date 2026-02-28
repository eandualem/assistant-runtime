"""Repo management tools -- onboard, check status, and list repositories."""

from __future__ import annotations

from typing import Any

from loguru import logger

from lovely_assistant.services.tools._backbone_client import backbone_error, backbone_request
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

_VALID_ORGS: frozenset[str] = frozenset({"Arclio", "WF", "Loveble"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_org(org: str) -> str | None:
    """Validate org parameter. Returns error message or None if valid."""
    if not org or not org.strip():
        return "Org cannot be empty"
    if org not in _VALID_ORGS:
        return f"Invalid org '{org}'. Must be one of: {', '.join(sorted(_VALID_ORGS))}"
    return None


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def onboard_repo(org: str, url: str) -> dict[str, Any]:
    """Onboard a new repository -- clone, configure CLAUDE.md, .claude/, settings, registry."""
    error = _validate_org(org)
    if error:
        return {"error": error, "success": False}

    if not url or not url.strip():
        return {"error": "URL cannot be empty", "success": False}

    status, data = await backbone_request(
        "POST",
        "/api/repos/onboard",
        json_body={"org": org, "url": url.strip()},
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status not in (200, 201):
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {
        "repo": data.get("repo"),
        "org": data.get("org", org),
        "steps": data.get("steps", []),
        "success": True,
    }


async def check_repo_status(org: str, repo: str) -> dict[str, Any]:
    """Check the onboarding/configuration status of a specific repository."""
    error = _validate_org(org)
    if error:
        return {"error": error, "success": False}

    if not repo or not repo.strip():
        return {"error": "Repo name cannot be empty", "success": False}

    status, data = await backbone_request(
        "GET",
        f"/api/repos/{org}/{repo.strip()}/status",
    )

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status == 404:
        return {"error": f"Repository {org}/{repo} not found", "success": False}

    if status != 200:
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    return {**data, "success": True}


async def list_repos() -> dict[str, Any]:
    """List all managed repositories across all orgs."""
    status, data = await backbone_request("GET", "/api/repos")

    if status == -1:
        return {"error": backbone_error(data), "success": False}

    if status != 200:
        return {
            "error": f"Backbone API error ({status}): {data.get('detail', 'Unknown error')}",
            "success": False,
        }

    repos = data.get("repos", data.get("items", []))
    return {"repos": repos, "count": len(repos), "success": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_repo_tools(registry: ToolRegistry) -> None:
    """Register all repo management tools."""
    registry.register_backend_tool(
        ToolDefinition(
            name="onboard_repo",
            description=(
                "Onboard a new repository into the workspace. Clones the repo, "
                "sets up CLAUDE.md, .claude/ directory, settings, and registers it "
                "in the agent backbone. Requires the org (Arclio, WF, or Loveble) "
                "and the Git URL."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "org": {
                        "type": "string",
                        "enum": ["Arclio", "WF", "Loveble"],
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
        onboard_repo,
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
                        "enum": ["Arclio", "WF", "Loveble"],
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
        check_repo_status,
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
        list_repos,
    )

    logger.info("Registered repo management tools", count=3)
