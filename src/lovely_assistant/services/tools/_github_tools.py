"""GitHub issue management tools -- create, search, read, comment, close issues."""

from __future__ import annotations

import os
from typing import Any

import httpx
from loguru import logger

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO_OWNER = "eandualem"
GITHUB_REPO_NAME = "orchestration"
GITHUB_API_BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _github_request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, str] | None = None,
) -> tuple[int, Any]:
    """Make a GitHub API request. Returns (status_code, parsed_json_body).

    Returns (-1, error_dict) if the token is not configured or on network error.
    """
    if not GITHUB_TOKEN:
        return (-1, {"message": "GITHUB_TOKEN not configured"})

    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    url = f"{GITHUB_API_BASE}{path}"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
            )
            return (response.status_code, response.json())
    except httpx.TimeoutException:
        return (-1, {"message": f"Request timed out: {method} {path}"})
    except httpx.HTTPError as exc:
        return (-1, {"message": f"HTTP error: {exc}"})


def _has_label_prefix(labels: list[str], prefix: str) -> bool:
    """Check if any label starts with the given prefix."""
    return any(label.startswith(prefix) for label in labels)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


async def create_issue(
    title: str,
    body: str,
    labels: list[str],
    priority: str = "",
) -> dict[str, Any]:
    """Create a new issue in the orchestration repo."""
    if not title or not title.strip():
        return {"error": "Title cannot be empty", "success": False}

    if not labels:
        return {
            "error": "Labels are required (must include from: and for: labels)",
            "success": False,
        }

    if not _has_label_prefix(labels, "from:"):
        return {"error": "Labels must include at least one 'from:' label", "success": False}

    if not _has_label_prefix(labels, "for:"):
        return {"error": "Labels must include at least one 'for:' label", "success": False}

    issue_labels = list(labels)
    if priority and priority in ("blocking", "non-blocking"):
        issue_labels.append(priority)

    status, data = await _github_request(
        "POST",
        f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues",
        json_body={"title": title.strip(), "body": body, "labels": issue_labels},
    )

    if status == -1:
        return {"error": data.get("message", "Request failed"), "success": False}

    if status not in (200, 201):
        return {
            "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
            "success": False,
        }

    return {
        "number": data.get("number"),
        "url": data.get("html_url"),
        "title": data.get("title"),
        "success": True,
    }


async def search_issues(
    state: str = "open",
    labels: list[str] | None = None,
    text: str = "",
    limit: int = 20,
) -> dict[str, Any]:
    """Search issues in the orchestration repo."""
    if text:
        # Use search endpoint for text queries
        query = f"{text} repo:{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME} is:issue state:{state}"
        if labels:
            for label in labels:
                query += f' label:"{label}"'

        status, data = await _github_request(
            "GET",
            "/search/issues",
            params={"q": query, "per_page": str(limit)},
        )

        if status == -1:
            return {"error": data.get("message", "Request failed"), "success": False}

        if status != 200:
            return {
                "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
                "success": False,
            }

        items = data.get("items", [])
    else:
        # Use list endpoint for simple filtering
        params: dict[str, str] = {
            "state": state,
            "per_page": str(limit),
        }
        if labels:
            params["labels"] = ",".join(labels)

        status, data = await _github_request(
            "GET",
            f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues",
            params=params,
        )

        if status == -1:
            return {"error": data.get("message", "Request failed"), "success": False}

        if status != 200:
            return {
                "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
                "success": False,
            }

        items = data if isinstance(data, list) else []

    issues = [
        {
            "number": item.get("number"),
            "title": item.get("title"),
            "state": item.get("state"),
            "labels": [lbl.get("name", "") for lbl in item.get("labels", [])],
            "created_at": item.get("created_at"),
        }
        for item in items
    ]

    return {"issues": issues, "count": len(issues), "success": True}


async def get_issue_details(issue_number: int) -> dict[str, Any]:
    """Get full details of a specific issue including comments."""
    if issue_number <= 0:
        return {"error": "Issue number must be positive", "success": False}

    path = f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}"

    status, data = await _github_request("GET", path)

    if status == -1:
        return {"error": data.get("message", "Request failed"), "success": False}

    if status == 404:
        return {"error": f"Issue #{issue_number} not found", "success": False}

    if status != 200:
        return {
            "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
            "success": False,
        }

    # Fetch comments
    comment_status, comment_data = await _github_request("GET", f"{path}/comments")
    comments = []
    if comment_status == 200 and isinstance(comment_data, list):
        comments = [
            {
                "author": c.get("user", {}).get("login", "unknown"),
                "body": c.get("body", ""),
                "created_at": c.get("created_at"),
            }
            for c in comment_data
        ]

    return {
        "number": data.get("number"),
        "title": data.get("title"),
        "state": data.get("state"),
        "body": data.get("body", ""),
        "labels": [lbl.get("name", "") for lbl in data.get("labels", [])],
        "comments": comments,
        "success": True,
    }


async def comment_on_issue(issue_number: int, body: str) -> dict[str, Any]:
    """Add a comment to an issue."""
    if not body or not body.strip():
        return {"error": "Comment body cannot be empty", "success": False}

    status, data = await _github_request(
        "POST",
        f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}/comments",
        json_body={"body": body.strip()},
    )

    if status == -1:
        return {"error": data.get("message", "Request failed"), "success": False}

    if status not in (200, 201):
        return {
            "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
            "success": False,
        }

    return {
        "issue_number": issue_number,
        "comment_id": data.get("id"),
        "success": True,
    }


async def close_issue(issue_number: int, comment: str = "") -> dict[str, Any]:
    """Close an issue, optionally adding a closing comment first."""
    comment_added = False

    if comment and comment.strip():
        c_status, c_data = await _github_request(
            "POST",
            f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}/comments",
            json_body={"body": comment.strip()},
        )
        if c_status == -1:
            return {"error": c_data.get("message", "Request failed"), "success": False}
        if c_status not in (200, 201):
            return {
                "error": f"Failed to add closing comment ({c_status}): {c_data.get('message', 'Unknown error')}",
                "success": False,
            }
        comment_added = True

    status, data = await _github_request(
        "PATCH",
        f"/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}",
        json_body={"state": "closed", "state_reason": "completed"},
    )

    if status == -1:
        return {"error": data.get("message", "Request failed"), "success": False}

    if status != 200:
        return {
            "error": f"GitHub API error ({status}): {data.get('message', 'Unknown error')}",
            "success": False,
        }

    return {
        "issue_number": issue_number,
        "closed": True,
        "comment_added": comment_added,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_github_tools(registry: ToolRegistry) -> None:
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
        create_issue,
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
        search_issues,
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
        get_issue_details,
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
        comment_on_issue,
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
        close_issue,
    )

    logger.info("Registered GitHub issue management tools", count=5)
