"""Tests for skill management tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._skill_tools import (
    _parse_frontmatter,
    _scan_skills_dir,
    list_skills,
    read_skill,
    register_skill_tools,
)
from lovely_assistant.services.tools.config import ToolConfig

MODULE = "lovely_assistant.services.tools._skill_tools"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def global_skills_dir(tmp_path, monkeypatch):
    """Redirect GLOBAL_SKILLS_PATH (and SKILLS_BASE_PATH alias) to a temp directory."""
    global_dir = tmp_path / "global_skills"
    global_dir.mkdir()
    monkeypatch.setattr(f"{MODULE}.GLOBAL_SKILLS_PATH", global_dir)
    monkeypatch.setattr(f"{MODULE}.SKILLS_BASE_PATH", global_dir)
    return global_dir


@pytest.fixture
def repo_skills_setup(tmp_path):
    """Create fake repo home directories with .claude/skills/ and return agent list."""
    repos: dict[str, Path] = {}

    def _make_repo(session: str) -> Path:
        home = tmp_path / session
        skills = home / ".claude" / "skills"
        skills.mkdir(parents=True)
        repos[session] = home
        return skills

    return _make_repo, repos


@pytest.fixture
def mock_registry_empty():
    """Mock agent registry returning no agents."""
    cache = AsyncMock()
    cache.get_agents = AsyncMock(return_value=[])
    with patch(f"{MODULE}.get_registry_cache", return_value=cache):
        yield cache


@pytest.fixture
def mock_registry(repo_skills_setup):
    """Mock agent registry returning agents with home dirs from repo_skills_setup."""
    _make_repo, repos = repo_skills_setup

    async def _get_agents():
        return [
            {"session": session, "home": str(home)}
            for session, home in repos.items()
        ]

    cache = AsyncMock()
    cache.get_agents = AsyncMock(side_effect=_get_agents)
    patcher = patch(f"{MODULE}.get_registry_cache", return_value=cache)
    patcher.start()
    yield _make_repo, repos, cache
    patcher.stop()


def _create_skill(skills_dir: Path, name: str, content: str | None = None) -> None:
    """Helper to create a skill directory with SKILL.md."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    if content is not None:
        (skill_dir / "SKILL.md").write_text(content)


# ---------------------------------------------------------------------------
# TestParseFrontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_valid_frontmatter(self):
        content = "---\nname: my-skill\ndescription: A test skill\n---\n\n# Content"
        result = _parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == "A test skill"

    def test_multiline_description(self):
        content = "---\nname: sdd\ndescription: |\n  Line one.\n  Line two.\n---\n\nBody"
        result = _parse_frontmatter(content)
        assert result["name"] == "sdd"
        assert "Line one." in result["description"]
        assert "Line two." in result["description"]

    def test_no_frontmatter(self):
        content = "# Just a heading\n\nNo frontmatter here."
        result = _parse_frontmatter(content)
        assert result["name"] is None
        assert result["description"] is None

    def test_unclosed_frontmatter(self):
        content = "---\nname: broken\n\nNo closing delimiter."
        result = _parse_frontmatter(content)
        assert result["name"] is None

    def test_invalid_yaml(self):
        content = "---\n: invalid: yaml: {{{\n---\n\nBody."
        result = _parse_frontmatter(content)
        assert result["name"] is None

    def test_non_dict_yaml(self):
        content = "---\n- list\n- items\n---\n\nBody."
        result = _parse_frontmatter(content)
        assert result["name"] is None

    def test_empty_frontmatter(self):
        content = "---\n---\n\nBody."
        result = _parse_frontmatter(content)
        assert result["name"] is None
        assert result["description"] is None


# ---------------------------------------------------------------------------
# TestScanSkillsDir
# ---------------------------------------------------------------------------


class TestScanSkillsDir:
    def test_nonexistent_dir(self, tmp_path):
        result = _scan_skills_dir(tmp_path / "nope", scope="global")
        assert result == []

    def test_global_scope_tagging(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        _create_skill(skills, "foo", "---\nname: foo\n---\n\nBody")

        result = _scan_skills_dir(skills, scope="global")
        assert len(result) == 1
        assert result[0]["scope"] == "global"
        assert "repo" not in result[0]

    def test_repo_scope_tagging(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        _create_skill(skills, "bar", "---\nname: bar\n---\n\nBody")

        result = _scan_skills_dir(skills, scope="repo", repo="my-repo")
        assert len(result) == 1
        assert result[0]["scope"] == "repo"
        assert result[0]["repo"] == "my-repo"

    def test_skips_dirs_without_skill_md(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "empty-dir").mkdir()

        result = _scan_skills_dir(skills, scope="global")
        assert result == []

    def test_skips_non_directory_entries(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "stray.txt").write_text("not a skill")
        _create_skill(skills, "real", "---\nname: real\n---\n\nBody")

        result = _scan_skills_dir(skills, scope="global")
        assert len(result) == 1
        assert result[0]["name"] == "real"


# ---------------------------------------------------------------------------
# TestListSkills — global only (no registry)
# ---------------------------------------------------------------------------


class TestListSkillsGlobal:
    async def test_empty_directory(self, global_skills_dir, mock_registry_empty):
        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_nonexistent_directory(self, tmp_path, monkeypatch, mock_registry_empty):
        monkeypatch.setattr(f"{MODULE}.GLOBAL_SKILLS_PATH", tmp_path / "nonexistent")
        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_lists_skills_with_frontmatter(self, global_skills_dir, mock_registry_empty):
        _create_skill(
            global_skills_dir,
            "my-skill",
            "---\nname: my-skill\ndescription: Does things\n---\n\n# Content",
        )
        _create_skill(
            global_skills_dir,
            "other-skill",
            "---\nname: other-skill\ndescription: Does other things\n---\n\nBody",
        )

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 2

        names = {s["name"] for s in result["skills"]}
        assert names == {"my-skill", "other-skill"}

    async def test_global_skills_have_global_scope(self, global_skills_dir, mock_registry_empty):
        _create_skill(global_skills_dir, "sdd", "---\nname: sdd\n---\n\nBody")

        result = await list_skills()
        assert result["skills"][0]["scope"] == "global"
        assert "repo" not in result["skills"][0]

    async def test_skill_without_frontmatter(self, global_skills_dir, mock_registry_empty):
        _create_skill(global_skills_dir, "plain-skill", "# No frontmatter\n\nJust content.")

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 1
        skill = result["skills"][0]
        assert skill["name"] == "plain-skill"  # falls back to dir name
        assert skill["description"] is None

    async def test_directory_without_skill_md(self, global_skills_dir, mock_registry_empty):
        """Directories without SKILL.md are skipped."""
        (global_skills_dir / "empty-dir").mkdir()

        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_non_directory_entries_ignored(self, global_skills_dir, mock_registry_empty):
        """Files directly in skills dir are ignored."""
        (global_skills_dir / "stray-file.txt").write_text("not a skill")
        _create_skill(global_skills_dir, "real-skill", "---\nname: real\n---\n\nBody")

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 1
        assert result["skills"][0]["name"] == "real"

    async def test_results_sorted_alphabetically(self, global_skills_dir, mock_registry_empty):
        _create_skill(global_skills_dir, "zebra", "---\nname: zebra\n---\n\nBody")
        _create_skill(global_skills_dir, "alpha", "---\nname: alpha\n---\n\nBody")
        _create_skill(global_skills_dir, "middle", "---\nname: middle\n---\n\nBody")

        result = await list_skills()
        names = [s["name"] for s in result["skills"]]
        assert names == ["alpha", "middle", "zebra"]

    async def test_includes_path(self, global_skills_dir, mock_registry_empty):
        _create_skill(global_skills_dir, "test-skill", "---\nname: test\n---\n\nBody")

        result = await list_skills()
        assert str(global_skills_dir / "test-skill") == result["skills"][0]["path"]


# ---------------------------------------------------------------------------
# TestListSkills — repo-level discovery
# ---------------------------------------------------------------------------


class TestListSkillsRepoLevel:
    async def test_discovers_repo_skills(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry
        repo_skills = _make_repo("agent-orchestration-dashboard")
        _create_skill(repo_skills, "governance-model", "---\nname: governance-model\n---\n\nBody")

        result = await list_skills()
        assert result["success"] is True
        repo_entries = [s for s in result["skills"] if s["scope"] == "repo"]
        assert len(repo_entries) == 1
        assert repo_entries[0]["name"] == "governance-model"
        assert repo_entries[0]["repo"] == "agent-orchestration-dashboard"

    async def test_combines_global_and_repo_skills(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry

        # Global skill
        _create_skill(global_skills_dir, "sdd-methodology", "---\nname: sdd-methodology\n---\n\nBody")

        # Repo skill
        repo_skills = _make_repo("my-repo")
        _create_skill(repo_skills, "local-skill", "---\nname: local-skill\n---\n\nBody")

        result = await list_skills()
        assert result["success"] is True
        names = {s["name"] for s in result["skills"]}
        assert names == {"sdd-methodology", "local-skill"}

        scopes = {s["name"]: s["scope"] for s in result["skills"]}
        assert scopes["sdd-methodology"] == "global"
        assert scopes["local-skill"] == "repo"

    async def test_repo_filter_excludes_global(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry

        _create_skill(global_skills_dir, "global-skill", "---\nname: global-skill\n---\n\nBody")
        repo_skills = _make_repo("dashboard")
        _create_skill(repo_skills, "local-skill", "---\nname: local-skill\n---\n\nBody")

        result = await list_skills(repo="dashboard")
        assert result["success"] is True
        assert len(result["skills"]) == 1
        assert result["skills"][0]["name"] == "local-skill"
        assert result["skills"][0]["scope"] == "repo"

    async def test_repo_filter_no_match(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry
        _make_repo("other-repo")

        result = await list_skills(repo="nonexistent")
        assert result["success"] is True
        assert result["skills"] == []

    async def test_multiple_repos(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry

        skills_a = _make_repo("repo-a")
        _create_skill(skills_a, "skill-a", "---\nname: skill-a\n---\n\nBody")

        skills_b = _make_repo("repo-b")
        _create_skill(skills_b, "skill-b", "---\nname: skill-b\n---\n\nBody")

        result = await list_skills()
        repo_skills = [s for s in result["skills"] if s["scope"] == "repo"]
        assert len(repo_skills) == 2
        repos_seen = {s["repo"] for s in repo_skills}
        assert repos_seen == {"repo-a", "repo-b"}

    async def test_skips_agents_without_home(self, global_skills_dir):
        """Agents missing home or session fields are skipped."""
        cache = AsyncMock()
        cache.get_agents = AsyncMock(return_value=[
            {"session": "no-home-agent"},  # missing home
            {"home": "/some/path"},  # missing session
        ])
        with patch(f"{MODULE}.get_registry_cache", return_value=cache):
            result = await list_skills()
            assert result["success"] is True
            # Only global skills (none created), no crash
            assert result["skills"] == []

    async def test_registry_unavailable(self, global_skills_dir):
        """If the registry returns None, only global skills are shown."""
        _create_skill(global_skills_dir, "global-one", "---\nname: global-one\n---\n\nBody")

        cache = AsyncMock()
        cache.get_agents = AsyncMock(return_value=None)
        with patch(f"{MODULE}.get_registry_cache", return_value=cache):
            result = await list_skills()
            assert result["success"] is True
            assert len(result["skills"]) == 1
            assert result["skills"][0]["scope"] == "global"


# ---------------------------------------------------------------------------
# TestReadSkill — global
# ---------------------------------------------------------------------------


class TestReadSkillGlobal:
    async def test_read_existing_skill(self, global_skills_dir, mock_registry_empty):
        content = "---\nname: my-skill\ndescription: Test\n---\n\n# Full Content\n\nBody here."
        _create_skill(global_skills_dir, "my-skill", content)

        result = await read_skill(name="my-skill")
        assert result["success"] is True
        assert result["name"] == "my-skill"
        assert result["content"] == content
        assert "repo" not in result  # global skill — no repo field

    async def test_skill_not_found(self, global_skills_dir, mock_registry_empty):
        result = await read_skill(name="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_empty_name(self, global_skills_dir):
        result = await read_skill(name="")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    async def test_path_traversal_dotdot(self, global_skills_dir):
        result = await read_skill(name="../etc")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_path_traversal_slash(self, global_skills_dir):
        result = await read_skill(name="foo/bar")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_path_traversal_backslash(self, global_skills_dir):
        result = await read_skill(name="foo\\bar")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_directory_exists_but_no_skill_md(self, global_skills_dir, mock_registry_empty):
        (global_skills_dir / "empty-skill").mkdir()
        result = await read_skill(name="empty-skill")
        assert result["success"] is False
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# TestReadSkill — repo-level
# ---------------------------------------------------------------------------


class TestReadSkillRepoLevel:
    async def test_read_repo_skill_by_name_and_repo(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry
        repo_skills = _make_repo("dashboard")
        content = "---\nname: governance-model\n---\n\n# Governance"
        _create_skill(repo_skills, "governance-model", content)

        result = await read_skill(name="governance-model", repo="dashboard")
        assert result["success"] is True
        assert result["name"] == "governance-model"
        assert result["repo"] == "dashboard"
        assert result["content"] == content

    async def test_repo_skill_not_found(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry
        _make_repo("dashboard")  # exists but has no skills

        result = await read_skill(name="nonexistent", repo="dashboard")
        assert result["success"] is False
        assert "not found" in result["error"]
        assert "dashboard" in result["error"]

    async def test_repo_not_in_registry(self, global_skills_dir, mock_registry):
        _make_repo, repos, _ = mock_registry

        result = await read_skill(name="something", repo="unknown-repo")
        assert result["success"] is False
        assert "not found in agent registry" in result["error"]

    async def test_path_traversal_on_repo_param(self, global_skills_dir):
        result = await read_skill(name="skill", repo="../etc")
        assert result["success"] is False
        assert "Invalid repo" in result["error"]

    async def test_path_traversal_slash_on_repo(self, global_skills_dir):
        result = await read_skill(name="skill", repo="foo/bar")
        assert result["success"] is False
        assert "Invalid repo" in result["error"]

    async def test_fallback_to_repo_when_not_in_global(self, global_skills_dir, mock_registry):
        """read_skill without repo= falls back to searching repo dirs."""
        _make_repo, repos, _ = mock_registry
        repo_skills = _make_repo("my-repo")
        content = "---\nname: local-only\n---\n\n# Local"
        _create_skill(repo_skills, "local-only", content)

        # No repo param — should find it via fallback
        result = await read_skill(name="local-only")
        assert result["success"] is True
        assert result["name"] == "local-only"
        assert result["repo"] == "my-repo"
        assert result["content"] == content

    async def test_global_takes_precedence_over_repo(self, global_skills_dir, mock_registry):
        """When a skill exists in both global and repo, global wins (no repo param)."""
        _make_repo, repos, _ = mock_registry
        repo_skills = _make_repo("my-repo")

        global_content = "---\nname: shared\n---\n\n# Global version"
        _create_skill(global_skills_dir, "shared", global_content)

        repo_content = "---\nname: shared\n---\n\n# Repo version"
        _create_skill(repo_skills, "shared", repo_content)

        result = await read_skill(name="shared")
        assert result["success"] is True
        assert result["content"] == global_content
        assert "repo" not in result  # global match

    async def test_repo_param_bypasses_global(self, global_skills_dir, mock_registry):
        """When repo= is specified, global is not checked."""
        _make_repo, repos, _ = mock_registry
        repo_skills = _make_repo("my-repo")

        _create_skill(global_skills_dir, "shared", "---\nname: shared\n---\n\n# Global")
        repo_content = "---\nname: shared\n---\n\n# Repo version"
        _create_skill(repo_skills, "shared", repo_content)

        result = await read_skill(name="shared", repo="my-repo")
        assert result["success"] is True
        assert result["content"] == repo_content
        assert result["repo"] == "my-repo"

    async def test_registry_unavailable_with_repo_param(self, global_skills_dir):
        """If registry is down and repo= is given, return clear error."""
        cache = AsyncMock()
        cache.get_agents = AsyncMock(return_value=None)
        with patch(f"{MODULE}.get_registry_cache", return_value=cache):
            result = await read_skill(name="skill", repo="some-repo")
            assert result["success"] is False
            assert "registry unavailable" in result["error"].lower()


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegisterSkillTools:
    def test_both_tools_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        names = registry.get_tool_names()
        assert "list_skills" in names
        assert "read_skill" in names

    def test_both_are_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        assert registry._backend_definitions["list_skills"].category == "backend"
        assert registry._backend_definitions["read_skill"].category == "backend"

    def test_handlers_are_callable(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        assert callable(registry._backend_handlers["list_skills"])
        assert callable(registry._backend_handlers["read_skill"])

    def test_read_skill_schema_requires_name(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        schema = registry._backend_definitions["read_skill"].parameters_schema
        assert "name" in schema["properties"]
        assert "name" in schema["required"]

    def test_read_skill_schema_has_repo(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        schema = registry._backend_definitions["read_skill"].parameters_schema
        assert "repo" in schema["properties"]

    def test_list_skills_schema_has_repo(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        schema = registry._backend_definitions["list_skills"].parameters_schema
        assert "repo" in schema["properties"]

    def test_list_skills_schema_no_required(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        schema = registry._backend_definitions["list_skills"].parameters_schema
        assert "required" not in schema or schema.get("required") == []
