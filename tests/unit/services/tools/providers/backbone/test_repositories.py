"""Tests for repo management tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.repositories import register_repositories_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.backbone.repositories import (
    BackboneRepositories,
    _validate_org,
    check_repo_status,
    list_repos,
    onboard_repo,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODULE = "assistant_runtime.services.tools.providers.backbone.repositories"


# ---------------------------------------------------------------------------
# TestValidateOrg
# ---------------------------------------------------------------------------


class TestValidateOrg:
    def test_any_org_valid_when_allowlist_unset(self, monkeypatch):
        monkeypatch.delenv("REPO_ORGS", raising=False)
        assert _validate_org("org-a") is None

    def test_valid_org_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("REPO_ORGS", "org-a, org-b")
        assert _validate_org("org-a") is None
        assert _validate_org("org-b") is None

    def test_invalid_org(self, monkeypatch):
        monkeypatch.setenv("REPO_ORGS", "org-a,org-b")
        result = _validate_org("InvalidOrg")
        assert result is not None
        assert "Invalid org" in result
        assert "org-a, org-b" in result

    def test_empty_org(self):
        result = _validate_org("")
        assert result is not None
        assert "empty" in result.lower()

    def test_whitespace_org(self):
        result = _validate_org("   ")
        assert result is not None
        assert "empty" in result.lower()


# ---------------------------------------------------------------------------
# TestOnboardRepo
# ---------------------------------------------------------------------------


class TestOnboardRepo:
    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "repo": "new-service",
                "org": "org-a",
                "steps": ["cloned", "claude_md", "settings", "registry"],
            },
        )

        result = await onboard_repo(
            org="org-a", url="https://github.com/example-org/new-service.git"
        )
        assert result["success"] is True
        assert result["repo"] == "new-service"
        assert result["org"] == "org-a"
        assert len(result["steps"]) == 4

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_backbone_error(self, mock_req):
        mock_req.return_value = (
            500,
            {"detail": "Clone failed"},
        )

        result = await onboard_repo(org="org-b", url="https://github.com/example-org/broken.git")
        assert result["success"] is False
        assert "500" in result["error"]

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_network_error(self, mock_req):
        mock_req.return_value = (
            -1,
            {"error": "Request timed out: POST /api/repos/onboard"},
        )

        result = await onboard_repo(org="org-a", url="https://github.com/example-org/repo.git")
        assert result["success"] is False
        assert "timed out" in result["error"]

    async def test_invalid_org(self, monkeypatch):
        monkeypatch.setenv("REPO_ORGS", "org-a,org-b")
        result = await onboard_repo(org="BadOrg", url="https://github.com/example-org/repo.git")
        assert result["success"] is False
        assert "Invalid org" in result["error"]

    async def test_empty_url(self):
        result = await onboard_repo(org="org-a", url="")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_whitespace_url(self):
        result = await onboard_repo(org="org-a", url="   ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_url_stripped(self, mock_req):
        mock_req.return_value = (200, {"repo": "r", "org": "org-b", "steps": []})

        await onboard_repo(org="org-b", url="  https://github.com/example-org/r.git  ")

        call_kwargs = mock_req.call_args
        json_body = call_kwargs.kwargs.get("json_body")
        assert json_body["url"] == "https://github.com/example-org/r.git"


# ---------------------------------------------------------------------------
# TestCheckRepoStatus
# ---------------------------------------------------------------------------


class TestCheckRepoStatus:
    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "repo": "platform-api",
                "org": "org-a",
                "status": "configured",
                "claude_md": True,
                "settings": True,
            },
        )

        result = await check_repo_status(org="org-a", repo="platform-api")
        assert result["success"] is True
        assert result["status"] == "configured"
        assert result["claude_md"] is True

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_not_found(self, mock_req):
        mock_req.return_value = (404, {"detail": "Not found"})

        result = await check_repo_status(org="org-a", repo="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_network_error(self, mock_req):
        mock_req.return_value = (
            -1,
            {"error": "HTTP error: connection refused"},
        )

        result = await check_repo_status(org="org-b", repo="some-repo")
        assert result["success"] is False
        assert "error" in result["error"].lower()

    async def test_invalid_org(self, monkeypatch):
        monkeypatch.setenv("REPO_ORGS", "org-a,org-b")
        result = await check_repo_status(org="BadOrg", repo="some-repo")
        assert result["success"] is False
        assert "Invalid org" in result["error"]

    async def test_empty_repo(self):
        result = await check_repo_status(org="org-a", repo="")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_repo_stripped(self, mock_req):
        mock_req.return_value = (200, {"repo": "r", "status": "ok"})

        await check_repo_status(org="org-b", repo="  my-repo  ")

        call_args = mock_req.call_args
        assert "/api/repos/org-b/my-repo/status" in call_args[0][1]


# ---------------------------------------------------------------------------
# TestListRepos
# ---------------------------------------------------------------------------


class TestListRepos:
    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_success(self, mock_req):
        mock_req.return_value = (
            200,
            {
                "repos": [
                    {"name": "platform-api", "org": "org-a"},
                    {"name": "agent-backbone", "org": "org-b"},
                ],
            },
        )

        result = await list_repos()
        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["repos"]) == 2

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_empty_list(self, mock_req):
        mock_req.return_value = (200, {"repos": []})

        result = await list_repos()
        assert result["success"] is True
        assert result["count"] == 0
        assert result["repos"] == []

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_network_error(self, mock_req):
        mock_req.return_value = (
            -1,
            {"error": "Request timed out: GET /api/repos"},
        )

        result = await list_repos()
        assert result["success"] is False
        assert "timed out" in result["error"]

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_server_error(self, mock_req):
        mock_req.return_value = (500, {"detail": "Internal Server Error"})

        result = await list_repos()
        assert result["success"] is False
        assert "500" in result["error"]

    @patch(f"{MODULE}.backbone_request", new_callable=AsyncMock)
    async def test_items_key_fallback(self, mock_req):
        """Backbone may return 'items' instead of 'repos'."""
        mock_req.return_value = (
            200,
            {"items": [{"name": "repo1", "org": "org-b"}]},
        )

        result = await list_repos()
        assert result["success"] is True
        assert result["count"] == 1


# ---------------------------------------------------------------------------
# TestRegisterRepoTools
# ---------------------------------------------------------------------------


class TestRegisterRepoTools:
    def test_all_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())

        names = registry.get_tool_names()
        assert "onboard_repo" in names
        assert "check_repo_status" in names
        assert "list_repos" in names

    def test_correct_count(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())
        assert len(registry._backend_definitions) == 3

    def test_all_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())
        for defn in registry._backend_definitions.values():
            assert defn.category == "backend"

    def test_schemas_have_required(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())

        must_require = {
            "onboard_repo": ["org", "url"],
            "check_repo_status": ["org", "repo"],
        }

        for tool_name, expected_required in must_require.items():
            defn = registry._backend_definitions[tool_name]
            schema_required = defn.parameters_schema.get("required", [])
            for field in expected_required:
                assert field in schema_required, f"{tool_name} missing required field '{field}'"

    def test_list_repos_no_required(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())
        defn = registry._backend_definitions["list_repos"]
        assert "required" not in defn.parameters_schema

    def test_handlers_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_repositories_tools(registry, BackboneRepositories())
        for name in ["onboard_repo", "check_repo_status", "list_repos"]:
            assert name in registry._backend_handlers
            assert callable(registry._backend_handlers[name])
