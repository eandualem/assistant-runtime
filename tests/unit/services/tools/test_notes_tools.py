"""Tests for notes management tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lovely_assistant.services.tools._notes_tools import (
    _build_note_content,
    _parse_note,
    _slugify,
    _validate_filename,
    manage_notes,
    register_notes_tools,
)
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.config import ToolConfig

MODULE = "lovely_assistant.services.tools._notes_tools"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_dir(tmp_path, monkeypatch):
    """Redirect NOTES_DIR to a temp directory."""
    monkeypatch.setattr(f"{MODULE}.NOTES_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# TestSlugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert _slugify("Auth flow: revisit!") == "auth-flow-revisit"

    def test_leading_trailing_stripped(self):
        assert _slugify("  --hello--  ") == "hello"

    def test_max_length(self):
        long_title = "a" * 200
        assert len(_slugify(long_title)) <= 80

    def test_empty(self):
        assert _slugify("") == ""

    def test_unicode(self):
        result = _slugify("caf\u00e9 latt\u00e9")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestValidateFilename
# ---------------------------------------------------------------------------


class TestValidateFilename:
    @pytest.mark.parametrize(
        "name",
        [
            "2026-02-19-hello-world.md",
            "my-note.md",
            "a.md",
            "test_note_123.md",
        ],
    )
    def test_valid(self, name):
        assert _validate_filename(name) is None

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("", "empty"),
            ("no-extension", "no .md extension"),
            ("../traversal.md", "path traversal"),
            ("has spaces.md", "spaces"),
            ("file.txt", "wrong extension"),
        ],
    )
    def test_invalid(self, name, reason):
        result = _validate_filename(name)
        assert result is not None, f"Expected invalid for: {reason}"


# ---------------------------------------------------------------------------
# TestParseNote
# ---------------------------------------------------------------------------


class TestParseNote:
    def test_with_frontmatter(self, notes_dir):
        path = notes_dir / "test.md"
        path.write_text(
            '---\ntitle: "My Note"\ndate: 2026-02-19\ntags: [auth, api]\n---\n\nNote body here.\n'
        )
        result = _parse_note(path)
        assert result is not None
        assert result["title"] == "My Note"
        assert result["date"] == "2026-02-19"
        assert result["tags"] == ["auth", "api"]
        assert result["content"] == "Note body here."

    def test_without_frontmatter(self, notes_dir):
        path = notes_dir / "plain.md"
        path.write_text("Just plain text.")
        result = _parse_note(path)
        assert result is not None
        assert result["title"] == "plain"  # stem as fallback
        assert result["content"] == "Just plain text."

    def test_missing_file(self, notes_dir):
        path = notes_dir / "nope.md"
        result = _parse_note(path)
        assert result is None

    def test_invalid_yaml(self, notes_dir):
        path = notes_dir / "bad.md"
        path.write_text("---\n: invalid: yaml: {{{\n---\n\nBody text.\n")
        result = _parse_note(path)
        assert result is not None
        # Should still parse with empty frontmatter
        assert result["content"] == "Body text."


# ---------------------------------------------------------------------------
# TestBuildNoteContent
# ---------------------------------------------------------------------------


class TestBuildNoteContent:
    @patch(f"{MODULE}.date")
    def test_basic(self, mock_date):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        content = _build_note_content("Test Title", "Body text", ["tag1", "tag2"])
        assert "---" in content
        assert "title: Test Title" in content
        assert "Body text" in content
        assert "tag1" in content

    @patch(f"{MODULE}.date")
    def test_no_tags(self, mock_date):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        content = _build_note_content("Title", "Body")
        assert "tags: []" in content


# ---------------------------------------------------------------------------
# TestManageNotes — create
# ---------------------------------------------------------------------------


class TestCreateNote:
    @patch(f"{MODULE}.date")
    async def test_create(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        mock_date.__format__ = lambda self, fmt: "2026-02-19"

        result = await manage_notes(
            action="create",
            title="Auth Flow Review",
            content="Need to revisit the auth flow next week.",
            tags=["auth", "next-week"],
        )
        assert result["success"] is True
        assert "auth-flow-review" in result["filename"]

        # Verify file was created
        path = notes_dir / result["filename"]
        assert path.exists()
        text = path.read_text()
        assert "Auth Flow Review" in text
        assert "revisit the auth flow" in text

    async def test_create_no_title(self, notes_dir):
        result = await manage_notes(action="create", content="Some content")
        assert result["success"] is False
        assert "Title" in result["error"]

    async def test_create_no_content(self, notes_dir):
        result = await manage_notes(action="create", title="Title")
        assert result["success"] is False
        assert "Content" in result["error"]

    @patch(f"{MODULE}.date")
    async def test_create_avoids_overwrite(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        mock_date.__format__ = lambda self, fmt: "2026-02-19"

        r1 = await manage_notes(action="create", title="Same Title", content="First")
        r2 = await manage_notes(action="create", title="Same Title", content="Second")
        assert r1["success"] is True
        assert r2["success"] is True
        assert r1["filename"] != r2["filename"]


# ---------------------------------------------------------------------------
# TestManageNotes — list
# ---------------------------------------------------------------------------


class TestListNotes:
    async def test_empty(self, notes_dir):
        result = await manage_notes(action="list")
        assert result["success"] is True
        assert result["count"] == 0

    async def test_with_notes(self, notes_dir):
        (notes_dir / "note1.md").write_text(
            '---\ntitle: "First"\ndate: 2026-02-18\ntags: [a]\n---\n\nFirst body.\n'
        )
        (notes_dir / "note2.md").write_text(
            '---\ntitle: "Second"\ndate: 2026-02-19\ntags: [b]\n---\n\nSecond body.\n'
        )

        result = await manage_notes(action="list")
        assert result["success"] is True
        assert result["count"] == 2

    async def test_filter_by_tag(self, notes_dir):
        (notes_dir / "a.md").write_text(
            '---\ntitle: "A"\ndate: 2026-02-19\ntags: [auth]\n---\n\nA body.\n'
        )
        (notes_dir / "b.md").write_text(
            '---\ntitle: "B"\ndate: 2026-02-19\ntags: [other]\n---\n\nB body.\n'
        )

        result = await manage_notes(action="list", tag="auth")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["notes"][0]["title"] == "A"

    async def test_limit(self, notes_dir):
        for i in range(5):
            (notes_dir / f"note{i}.md").write_text(
                f'---\ntitle: "Note {i}"\ndate: 2026-02-19\ntags: []\n---\n\nBody {i}.\n'
            )

        result = await manage_notes(action="list", limit=3)
        assert result["success"] is True
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# TestManageNotes — read
# ---------------------------------------------------------------------------


class TestReadNote:
    async def test_read(self, notes_dir):
        (notes_dir / "test.md").write_text(
            '---\ntitle: "Test"\ndate: 2026-02-19\ntags: [x]\n---\n\nFull content here.\n'
        )

        result = await manage_notes(action="read", filename="test.md")
        assert result["success"] is True
        assert result["title"] == "Test"
        assert result["content"] == "Full content here."

    async def test_read_not_found(self, notes_dir):
        result = await manage_notes(action="read", filename="nope.md")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_read_invalid_filename(self, notes_dir):
        result = await manage_notes(action="read", filename="../etc/passwd")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestManageNotes — search
# ---------------------------------------------------------------------------


class TestSearchNotes:
    async def test_search_in_content(self, notes_dir):
        (notes_dir / "note1.md").write_text(
            '---\ntitle: "Auth"\ndate: 2026-02-19\ntags: []\n---\n\nThe auth flow needs work.\n'
        )
        (notes_dir / "note2.md").write_text(
            '---\ntitle: "Other"\ndate: 2026-02-19\ntags: []\n---\n\nNothing relevant.\n'
        )

        result = await manage_notes(action="search", query="auth flow")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["filename"] == "note1.md"

    async def test_search_in_title(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Database Migration"\ndate: 2026-02-19\ntags: []\n---\n\nSome body.\n'
        )

        result = await manage_notes(action="search", query="database")
        assert result["success"] is True
        assert result["count"] == 1

    async def test_search_in_tags(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: [urgent]\n---\n\nSome body.\n'
        )

        result = await manage_notes(action="search", query="urgent")
        assert result["success"] is True
        assert result["count"] == 1

    async def test_search_no_query(self, notes_dir):
        result = await manage_notes(action="search")
        assert result["success"] is False
        assert "Query" in result["error"]

    async def test_search_no_results(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\nBody.\n'
        )

        result = await manage_notes(action="search", query="zzzznonexistent")
        assert result["success"] is True
        assert result["count"] == 0

    async def test_search_snippet_context(self, notes_dir):
        long_content = "x" * 100 + "MATCH_HERE" + "y" * 100
        (notes_dir / "note.md").write_text(
            f'---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\n{long_content}\n'
        )

        result = await manage_notes(action="search", query="MATCH_HERE")
        assert result["success"] is True
        snippet = result["results"][0]["snippet"]
        assert "MATCH_HERE" in snippet
        assert "..." in snippet  # truncation indicator


# ---------------------------------------------------------------------------
# TestManageNotes — update
# ---------------------------------------------------------------------------


class TestUpdateNote:
    async def test_update_content(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Original"\ndate: 2026-02-19\ntags: [old]\n---\n\nOld content.\n'
        )

        result = await manage_notes(action="update", filename="note.md", content="New content.")
        assert result["success"] is True

        text = (notes_dir / "note.md").read_text()
        assert "New content" in text
        assert "Original" in text  # title preserved
        assert "old" in text  # tags preserved

    async def test_update_tags(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: [old]\n---\n\nBody.\n'
        )

        result = await manage_notes(action="update", filename="note.md", tags=["new", "updated"])
        assert result["success"] is True

        text = (notes_dir / "note.md").read_text()
        assert "new" in text
        assert "Body." in text  # content preserved

    async def test_update_not_found(self, notes_dir):
        result = await manage_notes(action="update", filename="nope.md", content="x")
        assert result["success"] is False

    async def test_update_nothing_to_update(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\nBody.\n'
        )
        result = await manage_notes(action="update", filename="note.md")
        assert result["success"] is False
        assert "Provide" in result["error"]


# ---------------------------------------------------------------------------
# TestManageNotes — delete
# ---------------------------------------------------------------------------


class TestDeleteNote:
    async def test_delete(self, notes_dir):
        path = notes_dir / "doomed.md"
        path.write_text("content")

        result = await manage_notes(action="delete", filename="doomed.md")
        assert result["success"] is True
        assert result["deleted"] is True
        assert not path.exists()

    async def test_delete_not_found(self, notes_dir):
        result = await manage_notes(action="delete", filename="nope.md")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_delete_invalid_filename(self, notes_dir):
        result = await manage_notes(action="delete", filename="../bad.md")
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TestManageNotes — invalid action
# ---------------------------------------------------------------------------


class TestInvalidAction:
    async def test_unknown_action(self, notes_dir):
        result = await manage_notes(action="explode")
        assert result["success"] is False
        assert "Unknown action" in result["error"]


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegisterNotesTools:
    def test_tool_registered(self):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry)
        assert "manage_notes" in registry.get_tool_names()

    def test_is_backend(self):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry)
        assert registry._backend_definitions["manage_notes"].category == "backend"

    def test_has_handler(self):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry)
        assert callable(registry._backend_handlers["manage_notes"])

    def test_schema_has_action_enum(self):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry)
        schema = registry._backend_definitions["manage_notes"].parameters_schema
        assert "action" in schema["properties"]
        assert "enum" in schema["properties"]["action"]
        assert set(schema["properties"]["action"]["enum"]) == {
            "create",
            "list",
            "read",
            "search",
            "update",
            "delete",
        }
