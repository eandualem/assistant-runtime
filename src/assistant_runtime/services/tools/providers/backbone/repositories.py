"""Repo management tools -- onboard, check status, and list repositories."""

from __future__ import annotations

import os
from typing import Any

from assistant_runtime.services.tools.providers.backbone._client import (
    backbone_detail,
    backbone_error,
    backbone_request,
)

# Optional comma-separated allowlist of org names; unset means any non-empty org is accepted.
REPO_ORGS_ENV = "REPO_ORGS"


def _allowed_orgs() -> frozenset[str]:
    raw = os.environ.get(REPO_ORGS_ENV, "")
    return frozenset(o.strip() for o in raw.split(",") if o.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_org(org: str) -> str | None:
    """Validate org parameter. Returns error message or None if valid."""
    if not org or not org.strip():
        return "Org cannot be empty"
    allowed = _allowed_orgs()
    if allowed and org not in allowed:
        return f"Invalid org '{org}'. Must be one of: {', '.join(sorted(allowed))}"
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
            "error": f"Backbone API error ({status}): {backbone_detail(data)}",
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
            "error": f"Backbone API error ({status}): {backbone_detail(data)}",
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
            "error": f"Backbone API error ({status}): {backbone_detail(data)}",
            "success": False,
        }

    repos = data.get("repos", data.get("items", []))
    return {"repos": repos, "count": len(repos), "success": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class BackboneRepositories:
    """The repositories capability served by this provider (see ``capabilities.repositories``)."""

    async def onboard_repo(self, org: str, url: str) -> dict[str, Any]:
        return await onboard_repo(org=org, url=url)

    async def check_repo_status(self, org: str, repo: str) -> dict[str, Any]:
        return await check_repo_status(org=org, repo=repo)

    async def list_repos(self) -> dict[str, Any]:
        return await list_repos()
