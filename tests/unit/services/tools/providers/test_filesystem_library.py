"""Tests for the filesystem document library and the library capability."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.library import register_library_tools
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.filesystem import (
    FilesystemLibrary,
    parse_frontmatter,
)


def _document(root: Path, name: str, content: str | None = None, index: str = "SKILL.md") -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / index).write_text(
        content
        if content is not None
        else f"---\nname: {name}\ndescription: About {name}\n---\n# {name}\n"
    )


@pytest.fixture
def roots(tmp_path):
    global_root = tmp_path / "global"
    team_root = tmp_path / "team"
    global_root.mkdir()
    team_root.mkdir()
    _document(global_root, "deploy")
    _document(global_root, "review", content="no frontmatter here")
    _document(
        team_root, "deploy", content="---\nname: deploy\ndescription: Team deploy\n---\nteam\n"
    )
    _document(team_root, "onboarding", index="README.md")
    (global_root / "not-a-doc.md").write_text("stray file")
    (global_root / "empty").mkdir()
    return {"global": global_root, "team": team_root}


class TestParseFrontmatter:
    def test_reads_name_and_description(self):
        meta = parse_frontmatter("---\nname: x\ndescription: y\n---\nbody")
        assert meta == {"name": "x", "description": "y"}

    def test_missing_or_invalid_is_empty(self):
        assert parse_frontmatter("plain") == {}
        assert parse_frontmatter("---\nunclosed") == {}
        assert parse_frontmatter("---\n- not: [a mapping\n---\n") == {}


class TestFilesystemLibrary:
    async def test_lists_every_collection(self, roots):
        result = await FilesystemLibrary(roots).list_documents(collection=None)
        assert result["success"] is True
        names = {(d["collection"], d["name"]) for d in result["documents"]}
        assert names == {
            ("global", "deploy"),
            ("global", "review"),
            ("team", "deploy"),
            ("team", "onboarding"),
        }

    async def test_description_comes_from_frontmatter(self, roots):
        result = await FilesystemLibrary(roots).list_documents(collection="global")
        by_name = {d["name"]: d for d in result["documents"]}
        assert by_name["deploy"]["description"] == "About deploy"
        assert by_name["review"]["description"] is None

    async def test_unknown_collection_is_an_error(self, roots):
        result = await FilesystemLibrary(roots).list_documents(collection="nope")
        assert result["success"] is False

    async def test_missing_root_lists_nothing(self, tmp_path):
        result = await FilesystemLibrary({"x": tmp_path / "absent"}).list_documents(collection=None)
        assert result == {"success": True, "documents": []}

    async def test_reads_first_collection_that_has_the_name(self, roots):
        result = await FilesystemLibrary(roots).read_document(name="deploy", collection=None)
        assert result["collection"] == "global"
        assert result["content"].startswith("---\nname: deploy")

    async def test_reads_from_a_named_collection(self, roots):
        result = await FilesystemLibrary(roots).read_document(name="deploy", collection="team")
        assert result["collection"] == "team"
        assert "Team deploy" in result["content"]

    async def test_alternative_index_names(self, roots):
        result = await FilesystemLibrary(roots).read_document(name="onboarding", collection=None)
        assert result["success"] is True

    async def test_not_found_and_traversal(self, roots):
        library = FilesystemLibrary(roots)
        assert (await library.read_document(name="missing", collection=None))["success"] is False
        assert (await library.read_document(name="../deploy", collection=None))["success"] is False


class TestLibraryCapability:
    async def test_registers_two_tools_bound_to_the_library(self, roots):
        registry = ToolRegistry(ToolConfig())
        register_library_tools(registry, FilesystemLibrary(roots))
        assert set(registry.get_tool_names()) == {"list_documents", "read_document"}

        listed = await registry._backend_handlers["list_documents"](collection="team")
        assert {d["name"] for d in listed["documents"]} == {"deploy", "onboarding"}
        read = await registry._backend_handlers["read_document"](name="onboarding")
        assert read["success"] is True
        assert (await registry._backend_handlers["read_document"](name=""))["success"] is False
