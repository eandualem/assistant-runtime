"""Tests for GitHub issue management tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from lovely_assistant.services.tools._github_tools import (
    _github_request,
    _has_label_prefix,
    close_issue,
    comment_on_issue,
    create_issue,
    get_issue_details,
    register_github_tools,
    search_issues,
)
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.config import ToolConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODULE = "lovely_assistant.services.tools._github_tools"


# ---------------------------------------------------------------------------
# TestHasLabelPrefix
# ---------------------------------------------------------------------------


class TestHasLabelPrefix:
    def test_has_from_label(self):
        assert _has_label_prefix(["from:coding-agent", "for:ike", "task"], "from:") is True

    def test_has_for_label(self):
        assert _has_label_prefix(["from:coding-agent", "for:ike"], "for:") is True

    def test_missing_prefix(self):
        assert _has_label_prefix(["task", "bug"], "from:") is False

    def test_empty_labels(self):
        assert _has_label_prefix([], "from:") is False


# ---------------------------------------------------------------------------
# TestGithubRequest
# ---------------------------------------------------------------------------


class TestGithubRequest:
    @patch(f"{MODULE}.GITHUB_TOKEN", "")
    async def test_missing_token(self):
        status, data = await _github_request("GET", "/repos/test/test/issues")
        assert status == -1
        assert "not configured" in data["message"]

    @patch(f"{MODULE}.GITHUB_TOKEN", "ghp_test_token")
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_successful_get(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"number": 1}]

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await _github_request("GET", "/repos/test/test/issues")
        assert status == 200
        assert data == [{"number": 1}]

    @patch(f"{MODULE}.GITHUB_TOKEN", "ghp_test_token")
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_successful_post(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 42, "html_url": "https://github.com/test/42"}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await _github_request(
            "POST", "/repos/test/test/issues", json_body={"title": "test"}
        )
        assert status == 201
        assert data["number"] == 42

    @patch(f"{MODULE}.GITHUB_TOKEN", "ghp_test_token")
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_non_success_status(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {"message": "Validation Failed"}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await _github_request("POST", "/repos/test/test/issues")
        assert status == 422
        assert data["message"] == "Validation Failed"

    @patch(f"{MODULE}.GITHUB_TOKEN", "ghp_test_token")
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_timeout(self, mock_client_cls):
        import httpx

        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await _github_request("GET", "/repos/test/test/issues")
        assert status == -1
        assert "timed out" in data["message"]


# ---------------------------------------------------------------------------
# TestCreateIssue
# ---------------------------------------------------------------------------


class TestCreateIssue:
    @patch(f"{MODULE}._github_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (
            201,
            {
                "number": 99,
                "html_url": "https://github.com/eandualem/orchestration/issues/99",
                "title": "Test issue",
            },
        )

        result = await create_issue(
            title="[task] Test issue",
            body="## Context\nTest",
            labels=["from:coding-agent", "for:ike", "task"],
        )
        assert result["success"] is True
        assert result["number"] == 99
        assert result["url"] == "https://github.com/eandualem/orchestration/issues/99"
        assert result["title"] == "Test issue"

    async def test_missing_title(self):
        result = await create_issue(title="", body="body", labels=["from:x", "for:y"])
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_missing_from_label(self):
        result = await create_issue(title="Test", body="body", labels=["for:ike", "task"])
        assert result["success"] is False
        assert "from:" in result["error"]

    async def test_missing_for_label(self):
        result = await create_issue(title="Test", body="body", labels=["from:coding-agent", "task"])
        assert result["success"] is False
        assert "for:" in result["error"]

    async def test_empty_labels(self):
        result = await create_issue(title="Test", body="body", labels=[])
        assert result["success"] is False
        assert "required" in result["error"].lower()

    @patch(f"{MODULE}._github_request")
    async def test_priority_added(self, mock_req):
        mock_req.return_value = (201, {"number": 1, "html_url": "url", "title": "T"})

        await create_issue(
            title="Test",
            body="body",
            labels=["from:coding-agent", "for:ike"],
            priority="blocking",
        )

        call_kwargs = mock_req.call_args
        json_body = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert "blocking" in json_body["labels"]

    @patch(f"{MODULE}._github_request")
    async def test_invalid_priority_ignored(self, mock_req):
        mock_req.return_value = (201, {"number": 1, "html_url": "url", "title": "T"})

        await create_issue(
            title="Test",
            body="body",
            labels=["from:coding-agent", "for:ike"],
            priority="urgent",
        )

        call_kwargs = mock_req.call_args
        json_body = call_kwargs.kwargs.get("json_body") or call_kwargs[1].get("json_body")
        assert "urgent" not in json_body["labels"]

    @patch(f"{MODULE}._github_request")
    async def test_no_token(self, mock_req):
        mock_req.return_value = (-1, {"message": "GITHUB_TOKEN not configured"})

        result = await create_issue(title="Test", body="body", labels=["from:x", "for:y"])
        assert result["success"] is False
        assert "not configured" in result["error"]


# ---------------------------------------------------------------------------
# TestSearchIssues
# ---------------------------------------------------------------------------


class TestSearchIssues:
    @patch(f"{MODULE}._github_request")
    async def test_default_open(self, mock_req):
        mock_req.return_value = (
            200,
            [
                {
                    "number": 1,
                    "title": "Issue 1",
                    "state": "open",
                    "labels": [],
                    "created_at": "2026-01-01",
                },
            ],
        )

        result = await search_issues()
        assert result["success"] is True
        assert result["count"] == 1

        # Verify the list endpoint was called (not search)
        call_args = mock_req.call_args
        assert "/issues" in call_args[0][1]
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["state"] == "open"

    @patch(f"{MODULE}._github_request")
    async def test_with_labels(self, mock_req):
        mock_req.return_value = (200, [])

        await search_issues(labels=["for:ike", "task"])

        call_args = mock_req.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["labels"] == "for:ike,task"

    @patch(f"{MODULE}._github_request")
    async def test_with_text_query(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "items": [
                    {
                        "number": 5,
                        "title": "Found",
                        "state": "open",
                        "labels": [],
                        "created_at": "2026-01-01",
                    },
                ]
            },
        )

        result = await search_issues(text="authentication")
        assert result["success"] is True
        assert result["count"] == 1

        # Verify the search endpoint was used
        call_args = mock_req.call_args
        assert "/search/issues" in call_args[0][1]

    @patch(f"{MODULE}._github_request")
    async def test_limit(self, mock_req):
        mock_req.return_value = (200, [])

        await search_issues(limit=5)

        call_args = mock_req.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["per_page"] == "5"

    @patch(f"{MODULE}._github_request")
    async def test_formats_results(self, mock_req):
        mock_req.return_value = (
            200,
            [
                {
                    "number": 10,
                    "title": "Test Issue",
                    "state": "open",
                    "labels": [{"name": "task"}, {"name": "for:ike"}],
                    "created_at": "2026-02-01T00:00:00Z",
                },
            ],
        )

        result = await search_issues()
        issue = result["issues"][0]
        assert issue["number"] == 10
        assert issue["title"] == "Test Issue"
        assert issue["state"] == "open"
        assert "task" in issue["labels"]
        assert "for:ike" in issue["labels"]
        assert issue["created_at"] == "2026-02-01T00:00:00Z"


# ---------------------------------------------------------------------------
# TestGetIssueDetails
# ---------------------------------------------------------------------------


class TestGetIssueDetails:
    @patch(f"{MODULE}._github_request")
    async def test_success(self, mock_req):
        mock_req.side_effect = [
            (
                200,
                {
                    "number": 42,
                    "title": "Test",
                    "state": "open",
                    "body": "Issue body",
                    "labels": [{"name": "task"}],
                },
            ),
            (
                200,
                [
                    {"user": {"login": "ike"}, "body": "Working on it", "created_at": "2026-02-01"},
                ],
            ),
        ]

        result = await get_issue_details(42)
        assert result["success"] is True
        assert result["number"] == 42
        assert result["body"] == "Issue body"
        assert len(result["comments"]) == 1
        assert result["comments"][0]["author"] == "ike"

    @patch(f"{MODULE}._github_request")
    async def test_not_found(self, mock_req):
        mock_req.return_value = (404, {"message": "Not Found"})

        result = await get_issue_details(99999)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_invalid_number(self):
        result = await get_issue_details(0)
        assert result["success"] is False
        assert "positive" in result["error"]

    async def test_negative_number(self):
        result = await get_issue_details(-5)
        assert result["success"] is False
        assert "positive" in result["error"]


# ---------------------------------------------------------------------------
# TestCommentOnIssue
# ---------------------------------------------------------------------------


class TestCommentOnIssue:
    @patch(f"{MODULE}._github_request")
    async def test_success(self, mock_req):
        mock_req.return_value = (201, {"id": 12345})

        result = await comment_on_issue(42, "Acknowledged")
        assert result["success"] is True
        assert result["comment_id"] == 12345
        assert result["issue_number"] == 42

    async def test_empty_body(self):
        result = await comment_on_issue(42, "")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_whitespace_body(self):
        result = await comment_on_issue(42, "   ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @patch(f"{MODULE}._github_request")
    async def test_api_error(self, mock_req):
        mock_req.return_value = (422, {"message": "Validation Failed"})

        result = await comment_on_issue(42, "Comment text")
        assert result["success"] is False
        assert "422" in result["error"]


# ---------------------------------------------------------------------------
# TestCloseIssue
# ---------------------------------------------------------------------------


class TestCloseIssue:
    @patch(f"{MODULE}._github_request")
    async def test_close_without_comment(self, mock_req):
        mock_req.return_value = (200, {"state": "closed"})

        result = await close_issue(42)
        assert result["success"] is True
        assert result["closed"] is True
        assert result["comment_added"] is False

        # Only one call: the PATCH
        assert mock_req.call_count == 1
        assert mock_req.call_args[0][0] == "PATCH"

    @patch(f"{MODULE}._github_request")
    async def test_close_with_comment(self, mock_req):
        mock_req.side_effect = [
            (201, {"id": 100}),  # POST comment
            (200, {"state": "closed"}),  # PATCH close
        ]

        result = await close_issue(42, comment="Resolved in PR #50")
        assert result["success"] is True
        assert result["closed"] is True
        assert result["comment_added"] is True

        assert mock_req.call_count == 2
        assert mock_req.call_args_list[0][0][0] == "POST"
        assert mock_req.call_args_list[1][0][0] == "PATCH"

    @patch(f"{MODULE}._github_request")
    async def test_close_api_error(self, mock_req):
        mock_req.return_value = (500, {"message": "Internal Server Error"})

        result = await close_issue(42)
        assert result["success"] is False
        assert "500" in result["error"]

    @patch(f"{MODULE}._github_request")
    async def test_comment_failure_stops_close(self, mock_req):
        mock_req.return_value = (422, {"message": "Validation Failed"})

        result = await close_issue(42, comment="Closing")
        assert result["success"] is False
        # Should not attempt the PATCH since comment failed
        assert mock_req.call_count == 1


# ---------------------------------------------------------------------------
# TestRegisterGithubTools
# ---------------------------------------------------------------------------


class TestRegisterGithubTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_github_tools(registry)

        names = registry.get_tool_names()
        assert "create_issue" in names
        assert "search_issues" in names
        assert "get_issue_details" in names
        assert "comment_on_issue" in names
        assert "close_issue" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_github_tools(registry)
        assert len(registry._backend_definitions) == 5

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_github_tools(registry)
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_schemas_have_required(self):
        registry = ToolRegistry(ToolConfig())
        register_github_tools(registry)

        # Tools that must have required fields
        must_require = {
            "create_issue": ["title", "body", "labels"],
            "get_issue_details": ["issue_number"],
            "comment_on_issue": ["issue_number", "body"],
            "close_issue": ["issue_number"],
        }

        for tool_name, expected_required in must_require.items():
            defn = registry._backend_definitions[tool_name]
            schema_required = defn.parameters_schema.get("required", [])
            for field in expected_required:
                assert field in schema_required, f"{tool_name} missing required field '{field}'"

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_github_tools(registry)
        for name in [
            "create_issue",
            "search_issues",
            "get_issue_details",
            "comment_on_issue",
            "close_issue",
        ]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
