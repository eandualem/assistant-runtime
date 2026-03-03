"""Tests for skill management tools."""

from __future__ import annotations

import pytest

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._skill_tools import (
    _parse_frontmatter,
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
def skills_dir(tmp_path, monkeypatch):
    """Redirect SKILLS_BASE_PATH to a temp directory."""
    monkeypatch.setattr(f"{MODULE}.SKILLS_BASE_PATH", tmp_path)
    return tmp_path


def _create_skill(skills_dir, name: str, content: str | None = None) -> None:
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
# TestListSkills
# ---------------------------------------------------------------------------


class TestListSkills:
    async def test_empty_directory(self, skills_dir):
        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_nonexistent_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.SKILLS_BASE_PATH", tmp_path / "nonexistent")
        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_lists_skills_with_frontmatter(self, skills_dir):
        _create_skill(
            skills_dir,
            "my-skill",
            "---\nname: my-skill\ndescription: Does things\n---\n\n# Content",
        )
        _create_skill(
            skills_dir,
            "other-skill",
            "---\nname: other-skill\ndescription: Does other things\n---\n\nBody",
        )

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 2

        names = {s["name"] for s in result["skills"]}
        assert names == {"my-skill", "other-skill"}

    async def test_skill_without_frontmatter(self, skills_dir):
        _create_skill(skills_dir, "plain-skill", "# No frontmatter\n\nJust content.")

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 1
        skill = result["skills"][0]
        assert skill["name"] == "plain-skill"  # falls back to dir name
        assert skill["description"] is None

    async def test_directory_without_skill_md(self, skills_dir):
        """Directories without SKILL.md are skipped."""
        (skills_dir / "empty-dir").mkdir()

        result = await list_skills()
        assert result["success"] is True
        assert result["skills"] == []

    async def test_non_directory_entries_ignored(self, skills_dir):
        """Files directly in skills dir are ignored."""
        (skills_dir / "stray-file.txt").write_text("not a skill")
        _create_skill(skills_dir, "real-skill", "---\nname: real\n---\n\nBody")

        result = await list_skills()
        assert result["success"] is True
        assert len(result["skills"]) == 1
        assert result["skills"][0]["name"] == "real"

    async def test_results_sorted_alphabetically(self, skills_dir):
        _create_skill(skills_dir, "zebra", "---\nname: zebra\n---\n\nBody")
        _create_skill(skills_dir, "alpha", "---\nname: alpha\n---\n\nBody")
        _create_skill(skills_dir, "middle", "---\nname: middle\n---\n\nBody")

        result = await list_skills()
        names = [s["name"] for s in result["skills"]]
        assert names == ["alpha", "middle", "zebra"]

    async def test_includes_path(self, skills_dir):
        _create_skill(skills_dir, "test-skill", "---\nname: test\n---\n\nBody")

        result = await list_skills()
        assert str(skills_dir / "test-skill") == result["skills"][0]["path"]


# ---------------------------------------------------------------------------
# TestReadSkill
# ---------------------------------------------------------------------------


class TestReadSkill:
    async def test_read_existing_skill(self, skills_dir):
        content = "---\nname: my-skill\ndescription: Test\n---\n\n# Full Content\n\nBody here."
        _create_skill(skills_dir, "my-skill", content)

        result = await read_skill(name="my-skill")
        assert result["success"] is True
        assert result["name"] == "my-skill"
        assert result["content"] == content

    async def test_skill_not_found(self, skills_dir):
        result = await read_skill(name="nonexistent")
        assert result["success"] is False
        assert "not found" in result["error"]

    async def test_empty_name(self, skills_dir):
        result = await read_skill(name="")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    async def test_path_traversal_dotdot(self, skills_dir):
        result = await read_skill(name="../etc")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_path_traversal_slash(self, skills_dir):
        result = await read_skill(name="foo/bar")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_path_traversal_backslash(self, skills_dir):
        result = await read_skill(name="foo\\bar")
        assert result["success"] is False
        assert "Invalid" in result["error"]

    async def test_directory_exists_but_no_skill_md(self, skills_dir):
        (skills_dir / "empty-skill").mkdir()
        result = await read_skill(name="empty-skill")
        assert result["success"] is False
        assert "not found" in result["error"]


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

    def test_list_skills_schema_no_required(self):
        registry = ToolRegistry(ToolConfig())
        register_skill_tools(registry)
        schema = registry._backend_definitions["list_skills"].parameters_schema
        assert "required" not in schema or schema.get("required") == []
