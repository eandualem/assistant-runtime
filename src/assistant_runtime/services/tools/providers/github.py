"""GitHub issue management tools -- create, search, read, comment, close issues."""

from __future__ import annotations

import os
from typing import Any

import httpx

from assistant_runtime.base.resilience import retry_with_backoff

# Target repository for the issue tools, read from the environment at call time.
GITHUB_REPO_OWNER_ENV = "GITHUB_REPO_OWNER"
GITHUB_REPO_NAME_ENV = "GITHUB_REPO_NAME"


def _repo_slug() -> str:
    """Return "owner/name" for the configured issue repository."""
    owner = os.environ.get(GITHUB_REPO_OWNER_ENV, "")
    name = os.environ.get(GITHUB_REPO_NAME_ENV, "")
    return f"{owner}/{name}"


def _repo_config_error() -> dict[str, Any] | None:
    """Return an error dict when the target repository is not configured, else None."""
    missing = [
        env
        for env in (GITHUB_REPO_OWNER_ENV, GITHUB_REPO_NAME_ENV)
        if not os.environ.get(env, "").strip()
    ]
    if not missing:
        return None
    return {
        "success": False,
        "error": f"GitHub repository not configured. Set {' and '.join(missing)} in .env",
        "error_code": "GITHUB_REPO_MISSING",
    }


GITHUB_API_BASE = "https://api.github.com"
_GITHUB_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, ConnectionError, TimeoutError)


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
    Token is read at call time (not import time) so load_dotenv() in lifespan works.
    """
    github_token = os.environ.get("GITHUB_TOKEN", "")
    if not github_token:
        return (
            -1,
            {
                "success": False,
                "error": "GITHUB_TOKEN not configured. Set GITHUB_TOKEN in .env",
                "error_code": "GITHUB_AUTH_MISSING",
            },
        )

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json",
    }

    url = f"{GITHUB_API_BASE}{path}"

    @retry_with_backoff(
        max_attempts=3,
        min_wait=0.5,
        max_wait=10.0,
        retry_on=_GITHUB_RETRYABLE,
        name="github_request",
    )
    async def _request() -> tuple[int, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
            )
            return (response.status_code, response.json())

    try:
        return await _request()
    except httpx.TimeoutException:
        return (
            -1,
            {
                "success": False,
                "error": f"Request timed out: {method} {path}",
                "error_code": "GITHUB_TIMEOUT",
            },
        )
    except httpx.HTTPError as exc:
        return (
            -1,
            {
                "success": False,
                "error": f"HTTP error: {exc}",
                "error_code": "GITHUB_HTTP_ERROR",
            },
        )


def _request_error(payload: dict[str, Any]) -> str:
    """Extract normalized request error text from transport payload."""
    return payload.get("error", payload.get("message", "Request failed"))


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
    """Create a new issue in the configured repository."""
    config_error = _repo_config_error()
    if config_error:
        return config_error

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
        f"/repos/{_repo_slug()}/issues",
        json_body={"title": title.strip(), "body": body, "labels": issue_labels},
    )

    if status == -1:
        return {"error": _request_error(data), "success": False}

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
    config_error = _repo_config_error()
    if config_error:
        return config_error

    if text:
        # Use search endpoint for text queries
        query = f"{text} repo:{_repo_slug()} is:issue state:{state}"
        if labels:
            for label in labels:
                query += f' label:"{label}"'

        status, data = await _github_request(
            "GET",
            "/search/issues",
            params={"q": query, "per_page": str(limit)},
        )

        if status == -1:
            return {"error": _request_error(data), "success": False}

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
            f"/repos/{_repo_slug()}/issues",
            params=params,
        )

        if status == -1:
            return {"error": _request_error(data), "success": False}

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
    config_error = _repo_config_error()
    if config_error:
        return config_error

    if issue_number <= 0:
        return {"error": "Issue number must be positive", "success": False}

    path = f"/repos/{_repo_slug()}/issues/{issue_number}"

    status, data = await _github_request("GET", path)

    if status == -1:
        return {"error": _request_error(data), "success": False}

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
    config_error = _repo_config_error()
    if config_error:
        return config_error

    if not body or not body.strip():
        return {"error": "Comment body cannot be empty", "success": False}

    status, data = await _github_request(
        "POST",
        f"/repos/{_repo_slug()}/issues/{issue_number}/comments",
        json_body={"body": body.strip()},
    )

    if status == -1:
        return {"error": _request_error(data), "success": False}

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
    config_error = _repo_config_error()
    if config_error:
        return config_error

    comment_added = False

    if comment and comment.strip():
        c_status, c_data = await _github_request(
            "POST",
            f"/repos/{_repo_slug()}/issues/{issue_number}/comments",
            json_body={"body": comment.strip()},
        )
        if c_status == -1:
            return {"error": _request_error(c_data), "success": False}
        if c_status not in (200, 201):
            return {
                "error": f"Failed to add closing comment ({c_status}): {c_data.get('message', 'Unknown error')}",
                "success": False,
            }
        comment_added = True

    status, data = await _github_request(
        "PATCH",
        f"/repos/{_repo_slug()}/issues/{issue_number}",
        json_body={"state": "closed", "state_reason": "completed"},
    )

    if status == -1:
        return {"error": _request_error(data), "success": False}

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


class GitHubIssues:
    """The issues capability served by this provider (see ``capabilities.issues``)."""

    async def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str],
        priority: str = "",
    ) -> dict[str, Any]:
        return await create_issue(title=title, body=body, labels=labels, priority=priority)

    async def search_issues(
        self,
        state: str = "open",
        labels: list[str] | None = None,
        text: str = "",
        limit: int = 20,
    ) -> dict[str, Any]:
        return await search_issues(state=state, labels=labels, text=text, limit=limit)

    async def get_issue_details(self, issue_number: int) -> dict[str, Any]:
        return await get_issue_details(issue_number=issue_number)

    async def comment_on_issue(self, issue_number: int, body: str) -> dict[str, Any]:
        return await comment_on_issue(issue_number=issue_number, body=body)

    async def close_issue(self, issue_number: int, comment: str = "") -> dict[str, Any]:
        return await close_issue(issue_number=issue_number, comment=comment)
