"""Tests for notes management tool."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from assistant_runtime.services.tools._notes_tools import (
    _build_note_content,
    _parse_note,
    _slugify,
    _validate_filename,
    _validate_folder,
    _validate_note_path,
    manage_notes,
    register_notes_tools,
)
from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.config import ToolConfig

MODULE = "assistant_runtime.services.tools._notes_tools"


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
# TestValidateNotePath
# ---------------------------------------------------------------------------


class TestValidateNotePath:
    def test_valid_path_with_folder(self, notes_dir):
        assert _validate_note_path("governance/tracks/my-note.md") is None

    def test_valid_bare_filename(self, notes_dir):
        assert _validate_note_path("my-note.md") is None

    def test_deeply_nested(self, notes_dir):
        assert _validate_note_path("a/b/c/note.md") is None

    def test_empty(self):
        assert _validate_note_path("") is not None

    def test_path_traversal_dotdot(self):
        result = _validate_note_path("../outside/note.md")
        assert result is not None
        assert "traversal" in result.lower()

    def test_backslash_rejected(self):
        result = _validate_note_path("folder\\note.md")
        assert result is not None

    def test_invalid_filename_in_path(self, notes_dir):
        result = _validate_note_path("folder/has spaces.md")
        assert result is not None


# ---------------------------------------------------------------------------
# TestValidateFolder
# ---------------------------------------------------------------------------


class TestValidateFolder:
    def test_valid_single(self, notes_dir):
        assert _validate_folder("governance") is None

    def test_valid_nested(self, notes_dir):
        assert _validate_folder("governance/tracks") is None

    def test_empty(self):
        result = _validate_folder("")
        assert result is not None

    def test_dotdot(self):
        result = _validate_folder("../outside")
        assert result is not None
        assert "traversal" in result.lower()

    def test_backslash(self):
        result = _validate_folder("a\\b")
        assert result is not None


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
        assert result["folder"] == ""

    def test_with_folder(self, notes_dir):
        sub = notes_dir / "governance" / "tracks"
        sub.mkdir(parents=True)
        path = sub / "design.md"
        path.write_text('---\ntitle: "Design"\ndate: 2026-03-23\ntags: []\n---\n\nDesign doc.\n')

        result = _parse_note(path)
        assert result is not None
        assert result["folder"] == "governance/tracks"
        assert result["path"] == "governance/tracks/design.md"

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

    @patch(f"{MODULE}.date")
    async def test_create_in_folder(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-03-23"
        mock_date.__format__ = lambda self, fmt: "2026-03-23"

        result = await manage_notes(
            action="create",
            title="Design Doc",
            content="Governance track design.",
            folder="governance/tracks",
        )
        assert result["success"] is True
        assert "path" in result
        assert "governance/tracks" in result["path"]

        # Verify file exists in subfolder
        path = notes_dir / "governance" / "tracks" / result["filename"]
        assert path.exists()

    @patch(f"{MODULE}.date")
    async def test_create_in_folder_creates_parents(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-03-23"
        mock_date.__format__ = lambda self, fmt: "2026-03-23"

        result = await manage_notes(
            action="create",
            title="Deep Note",
            content="Body.",
            folder="a/b/c",
        )
        assert result["success"] is True
        assert (notes_dir / "a" / "b" / "c").is_dir()

    async def test_create_in_folder_traversal_rejected(self, notes_dir):
        result = await manage_notes(
            action="create",
            title="Bad",
            content="Body.",
            folder="../outside",
        )
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    @patch(f"{MODULE}.date")
    async def test_create_returns_path(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-03-23"
        mock_date.__format__ = lambda self, fmt: "2026-03-23"

        result = await manage_notes(action="create", title="Root Note", content="Body.")
        assert result["success"] is True
        assert "path" in result


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

    async def test_lists_notes_in_subdirectories(self, notes_dir):
        """rglob finds notes in subfolders."""
        sub = notes_dir / "governance"
        sub.mkdir()
        (sub / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nTrack body.\n'
        )
        (notes_dir / "root.md").write_text(
            '---\ntitle: "Root"\ndate: 2026-03-23\ntags: []\n---\n\nRoot body.\n'
        )

        result = await manage_notes(action="list")
        assert result["success"] is True
        assert result["count"] == 2
        titles = {n["title"] for n in result["notes"]}
        assert titles == {"Track", "Root"}

    async def test_list_includes_folder_field(self, notes_dir):
        sub = notes_dir / "docs"
        sub.mkdir()
        (sub / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-03-23\ntags: []\n---\n\nBody.\n'
        )

        result = await manage_notes(action="list")
        assert result["success"] is True
        note = result["notes"][0]
        assert note["folder"] == "docs"
        assert note["path"] == "docs/note.md"

    async def test_list_scoped_to_folder(self, notes_dir):
        """folder param scopes listing to a subdirectory."""
        gov = notes_dir / "governance"
        gov.mkdir()
        (gov / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nBody.\n'
        )
        (notes_dir / "root.md").write_text(
            '---\ntitle: "Root"\ndate: 2026-03-23\ntags: []\n---\n\nBody.\n'
        )

        result = await manage_notes(action="list", folder="governance")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["notes"][0]["title"] == "Track"

    async def test_list_nonexistent_folder(self, notes_dir):
        result = await manage_notes(action="list", folder="nonexistent")
        assert result["success"] is True
        assert result["count"] == 0


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

    async def test_read_with_path(self, notes_dir):
        """Can read notes in subfolders via path."""
        sub = notes_dir / "governance"
        sub.mkdir()
        (sub / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nTrack content.\n'
        )

        result = await manage_notes(action="read", filename="governance/track.md")
        assert result["success"] is True
        assert result["title"] == "Track"
        assert result["content"] == "Track content."
        assert result["folder"] == "governance"

    async def test_read_with_path_not_found(self, notes_dir):
        (notes_dir / "governance").mkdir()
        result = await manage_notes(action="read", filename="governance/nope.md")
        assert result["success"] is False
        assert "not found" in result["error"].lower()


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

    async def test_search_finds_notes_in_subdirs(self, notes_dir):
        sub = notes_dir / "governance"
        sub.mkdir()
        (sub / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nGovernance track design.\n'
        )

        result = await manage_notes(action="search", query="governance track")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["folder"] == "governance"

    async def test_search_scoped_to_folder(self, notes_dir):
        gov = notes_dir / "governance"
        gov.mkdir()
        (gov / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nDesign doc.\n'
        )
        (notes_dir / "root.md").write_text(
            '---\ntitle: "Root"\ndate: 2026-03-23\ntags: []\n---\n\nDesign doc.\n'
        )

        result = await manage_notes(action="search", query="design", folder="governance")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Track"


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

    async def test_update_note_in_subfolder(self, notes_dir):
        sub = notes_dir / "docs"
        sub.mkdir()
        (sub / "note.md").write_text(
            '---\ntitle: "Doc"\ndate: 2026-03-23\ntags: []\n---\n\nOld body.\n'
        )

        result = await manage_notes(action="update", filename="docs/note.md", content="New body.")
        assert result["success"] is True

        text = (sub / "note.md").read_text()
        assert "New body" in text
        assert "Doc" in text  # title preserved


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

    async def test_delete_note_in_subfolder(self, notes_dir):
        sub = notes_dir / "docs"
        sub.mkdir()
        path = sub / "doomed.md"
        path.write_text("content")

        result = await manage_notes(action="delete", filename="docs/doomed.md")
        assert result["success"] is True
        assert result["deleted"] is True
        assert not path.exists()


# ---------------------------------------------------------------------------
# TestManageNotes — create_folder
# ---------------------------------------------------------------------------


class TestCreateFolder:
    async def test_create_folder(self, notes_dir):
        result = await manage_notes(action="create_folder", folder="governance/tracks")
        assert result["success"] is True
        assert result["created"] is True
        assert (notes_dir / "governance" / "tracks").is_dir()

    async def test_create_folder_already_exists(self, notes_dir):
        (notes_dir / "existing").mkdir()
        result = await manage_notes(action="create_folder", folder="existing")
        assert result["success"] is True
        assert result["created"] is False
        assert "already exists" in result["message"].lower()

    async def test_create_folder_no_folder(self, notes_dir):
        result = await manage_notes(action="create_folder")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    async def test_create_folder_traversal(self, notes_dir):
        result = await manage_notes(action="create_folder", folder="../outside")
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    async def test_create_nested_folder(self, notes_dir):
        result = await manage_notes(action="create_folder", folder="a/b/c/d")
        assert result["success"] is True
        assert (notes_dir / "a" / "b" / "c" / "d").is_dir()


# ---------------------------------------------------------------------------
# TestManageNotes — move_note
# ---------------------------------------------------------------------------


class TestMoveNote:
    async def test_move_note(self, notes_dir):
        (notes_dir / "note.md").write_text("content")
        (notes_dir / "target").mkdir()

        result = await manage_notes(action="move_note", filename="note.md", folder="target")
        assert result["success"] is True
        assert not (notes_dir / "note.md").exists()
        assert (notes_dir / "target" / "note.md").exists()
        assert result["to"] == "target/note.md"

    async def test_move_creates_destination(self, notes_dir):
        (notes_dir / "note.md").write_text("content")

        result = await manage_notes(action="move_note", filename="note.md", folder="new/sub/dir")
        assert result["success"] is True
        assert (notes_dir / "new" / "sub" / "dir" / "note.md").exists()

    async def test_move_no_filename(self, notes_dir):
        result = await manage_notes(action="move_note", folder="target")
        assert result["success"] is False
        assert "filename" in result["error"].lower()

    async def test_move_no_folder(self, notes_dir):
        result = await manage_notes(action="move_note", filename="note.md")
        assert result["success"] is False
        assert "folder" in result["error"].lower()

    async def test_move_source_not_found(self, notes_dir):
        result = await manage_notes(action="move_note", filename="nope.md", folder="target")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_move_destination_conflict(self, notes_dir):
        (notes_dir / "note.md").write_text("original")
        target = notes_dir / "target"
        target.mkdir()
        (target / "note.md").write_text("existing")

        result = await manage_notes(action="move_note", filename="note.md", folder="target")
        assert result["success"] is False
        assert "already exists" in result["error"].lower()

    async def test_move_traversal_on_folder(self, notes_dir):
        (notes_dir / "note.md").write_text("content")
        result = await manage_notes(action="move_note", filename="note.md", folder="../outside")
        assert result["success"] is False

    async def test_move_from_subfolder(self, notes_dir):
        """Can move a note from a subfolder to another subfolder."""
        src = notes_dir / "old"
        src.mkdir()
        (src / "note.md").write_text("content")

        result = await manage_notes(action="move_note", filename="old/note.md", folder="new")
        assert result["success"] is True
        assert not (src / "note.md").exists()
        assert (notes_dir / "new" / "note.md").exists()


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
            "create_folder",
            "move_note",
        }

    def test_schema_has_folder_property(self):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry)
        schema = registry._backend_definitions["manage_notes"].parameters_schema
        assert "folder" in schema["properties"]
