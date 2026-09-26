"""Tests for the markdown notes provider and the notes capability."""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import stat
import threading
import time
from unittest.mock import patch

import pytest

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities.notes import (
    build_manage_notes,
    register_notes_tools,
)
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.filesystem import (
    MarkdownNotes,
    build_note_content,
    slugify,
    validate_filename,
)

MODULE = "assistant_runtime.services.tools.providers.filesystem"


def _store(root):
    return MarkdownNotes(root)


def _manage(root):
    return build_manage_notes(MarkdownNotes(root))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def notes_dir(tmp_path):
    """The root of a fresh notes store."""
    return tmp_path


# ---------------------------------------------------------------------------
# TestSlugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Auth flow: revisit!") == "auth-flow-revisit"

    def test_leading_trailing_stripped(self):
        assert slugify("  --hello--  ") == "hello"

    def test_max_length(self):
        long_title = "a" * 200
        assert len(slugify(long_title)) <= 80

    def test_empty(self):
        assert slugify("") == ""

    def test_unicode(self):
        result = slugify("caf\u00e9 latt\u00e9")
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
        assert validate_filename(name) is None

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
        result = validate_filename(name)
        assert result is not None, f"Expected invalid for: {reason}"


# ---------------------------------------------------------------------------
# TestValidateNotePath
# ---------------------------------------------------------------------------


class TestValidateNotePath:
    def test_valid_path_with_folder(self, notes_dir):
        assert _store(notes_dir).validate_note_path("governance/tracks/my-note.md") is None

    def test_valid_bare_filename(self, notes_dir):
        assert _store(notes_dir).validate_note_path("my-note.md") is None

    def test_deeply_nested(self, notes_dir):
        assert _store(notes_dir).validate_note_path("a/b/c/note.md") is None

    def test_empty(self, notes_dir):
        assert _store(notes_dir).validate_note_path("") is not None

    def test_path_traversal_dotdot(self, notes_dir):
        result = _store(notes_dir).validate_note_path("../outside/note.md")
        assert result is not None
        assert "traversal" in result.lower()

    def test_backslash_rejected(self, notes_dir):
        result = _store(notes_dir).validate_note_path("folder\\note.md")
        assert result is not None

    def test_invalid_filename_in_path(self, notes_dir):
        result = _store(notes_dir).validate_note_path("folder/has spaces.md")
        assert result is not None


# ---------------------------------------------------------------------------
# TestValidateFolder
# ---------------------------------------------------------------------------


class TestValidateFolder:
    def test_valid_single(self, notes_dir):
        assert _store(notes_dir).validate_folder("governance") is None

    def test_valid_nested(self, notes_dir):
        assert _store(notes_dir).validate_folder("governance/tracks") is None

    def test_empty(self, notes_dir):
        result = _store(notes_dir).validate_folder("")
        assert result is not None

    def test_dotdot(self, notes_dir):
        result = _store(notes_dir).validate_folder("../outside")
        assert result is not None
        assert "traversal" in result.lower()

    def test_backslash(self):
        result = _store(notes_dir).validate_folder("a\\b")
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
        result = _store(notes_dir).parse_note(path)
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

        result = _store(notes_dir).parse_note(path)
        assert result is not None
        assert result["folder"] == "governance/tracks"
        assert result["path"] == "governance/tracks/design.md"

    def test_without_frontmatter(self, notes_dir):
        path = notes_dir / "plain.md"
        path.write_text("Just plain text.")
        result = _store(notes_dir).parse_note(path)
        assert result is not None
        assert result["title"] == "plain"  # stem as fallback
        assert result["content"] == "Just plain text."

    def test_missing_file(self, notes_dir):
        path = notes_dir / "nope.md"
        result = _store(notes_dir).parse_note(path)
        assert result is None

    def test_invalid_yaml(self, notes_dir):
        path = notes_dir / "bad.md"
        path.write_text("---\n: invalid: yaml: {{{\n---\n\nBody text.\n")
        result = _store(notes_dir).parse_note(path)
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
        content = build_note_content("Test Title", "Body text", ["tag1", "tag2"])
        assert "---" in content
        assert "title: Test Title" in content
        assert "Body text" in content
        assert "tag1" in content

    @patch(f"{MODULE}.date")
    def test_no_tags(self, mock_date):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        content = build_note_content("Title", "Body")
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

        result = await _manage(notes_dir)(
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
        result = await _manage(notes_dir)(action="create", content="Some content")
        assert result["success"] is False
        assert "Title" in result["error"]

    async def test_create_no_content(self, notes_dir):
        result = await _manage(notes_dir)(action="create", title="Title")
        assert result["success"] is False
        assert "Content" in result["error"]

    @patch(f"{MODULE}.date")
    async def test_create_avoids_overwrite(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-02-19"
        mock_date.__format__ = lambda self, fmt: "2026-02-19"

        r1 = await _manage(notes_dir)(action="create", title="Same Title", content="First")
        r2 = await _manage(notes_dir)(action="create", title="Same Title", content="Second")
        assert r1["success"] is True
        assert r2["success"] is True
        assert r1["filename"] != r2["filename"]

    @patch(f"{MODULE}.date")
    async def test_create_in_folder(self, mock_date, notes_dir):
        mock_date.today.return_value = mock_date
        mock_date.__str__ = lambda self: "2026-03-23"
        mock_date.__format__ = lambda self, fmt: "2026-03-23"

        result = await _manage(notes_dir)(
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

        result = await _manage(notes_dir)(
            action="create",
            title="Deep Note",
            content="Body.",
            folder="a/b/c",
        )
        assert result["success"] is True
        assert (notes_dir / "a" / "b" / "c").is_dir()

    async def test_create_in_folder_traversal_rejected(self, notes_dir):
        result = await _manage(notes_dir)(
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

        result = await _manage(notes_dir)(action="create", title="Root Note", content="Body.")
        assert result["success"] is True
        assert "path" in result


# ---------------------------------------------------------------------------
# TestManageNotes — list
# ---------------------------------------------------------------------------


class TestListNotes:
    async def test_empty(self, notes_dir):
        result = await _manage(notes_dir)(action="list")
        assert result["success"] is True
        assert result["count"] == 0

    async def test_with_notes(self, notes_dir):
        (notes_dir / "note1.md").write_text(
            '---\ntitle: "First"\ndate: 2026-02-18\ntags: [a]\n---\n\nFirst body.\n'
        )
        (notes_dir / "note2.md").write_text(
            '---\ntitle: "Second"\ndate: 2026-02-19\ntags: [b]\n---\n\nSecond body.\n'
        )

        result = await _manage(notes_dir)(action="list")
        assert result["success"] is True
        assert result["count"] == 2

    async def test_filter_by_tag(self, notes_dir):
        (notes_dir / "a.md").write_text(
            '---\ntitle: "A"\ndate: 2026-02-19\ntags: [auth]\n---\n\nA body.\n'
        )
        (notes_dir / "b.md").write_text(
            '---\ntitle: "B"\ndate: 2026-02-19\ntags: [other]\n---\n\nB body.\n'
        )

        result = await _manage(notes_dir)(action="list", tag="auth")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["notes"][0]["title"] == "A"

    async def test_limit(self, notes_dir):
        for i in range(5):
            (notes_dir / f"note{i}.md").write_text(
                f'---\ntitle: "Note {i}"\ndate: 2026-02-19\ntags: []\n---\n\nBody {i}.\n'
            )

        result = await _manage(notes_dir)(action="list", limit=3)
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

        result = await _manage(notes_dir)(action="list")
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

        result = await _manage(notes_dir)(action="list")
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

        result = await _manage(notes_dir)(action="list", folder="governance")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["notes"][0]["title"] == "Track"

    async def test_list_nonexistent_folder(self, notes_dir):
        result = await _manage(notes_dir)(action="list", folder="nonexistent")
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

        result = await _manage(notes_dir)(action="read", filename="test.md")
        assert result["success"] is True
        assert result["title"] == "Test"
        assert result["content"] == "Full content here."

    async def test_read_not_found(self, notes_dir):
        result = await _manage(notes_dir)(action="read", filename="nope.md")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_read_invalid_filename(self, notes_dir):
        result = await _manage(notes_dir)(action="read", filename="../etc/passwd")
        assert result["success"] is False

    async def test_read_with_path(self, notes_dir):
        """Can read notes in subfolders via path."""
        sub = notes_dir / "governance"
        sub.mkdir()
        (sub / "track.md").write_text(
            '---\ntitle: "Track"\ndate: 2026-03-23\ntags: []\n---\n\nTrack content.\n'
        )

        result = await _manage(notes_dir)(action="read", filename="governance/track.md")
        assert result["success"] is True
        assert result["title"] == "Track"
        assert result["content"] == "Track content."
        assert result["folder"] == "governance"

    async def test_read_with_path_not_found(self, notes_dir):
        (notes_dir / "governance").mkdir()
        result = await _manage(notes_dir)(action="read", filename="governance/nope.md")
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

        result = await _manage(notes_dir)(action="search", query="auth flow")
        assert result["success"] is True
        assert result["count"] == 1
        assert result["results"][0]["filename"] == "note1.md"

    async def test_search_in_title(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Database Migration"\ndate: 2026-02-19\ntags: []\n---\n\nSome body.\n'
        )

        result = await _manage(notes_dir)(action="search", query="database")
        assert result["success"] is True
        assert result["count"] == 1

    async def test_search_in_tags(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: [urgent]\n---\n\nSome body.\n'
        )

        result = await _manage(notes_dir)(action="search", query="urgent")
        assert result["success"] is True
        assert result["count"] == 1

    async def test_search_no_query(self, notes_dir):
        result = await _manage(notes_dir)(action="search")
        assert result["success"] is False
        assert "Query" in result["error"]

    async def test_search_no_results(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\nBody.\n'
        )

        result = await _manage(notes_dir)(action="search", query="zzzznonexistent")
        assert result["success"] is True
        assert result["count"] == 0

    async def test_search_snippet_context(self, notes_dir):
        long_content = "x" * 100 + "MATCH_HERE" + "y" * 100
        (notes_dir / "note.md").write_text(
            f'---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\n{long_content}\n'
        )

        result = await _manage(notes_dir)(action="search", query="MATCH_HERE")
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

        result = await _manage(notes_dir)(action="search", query="governance track")
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

        result = await _manage(notes_dir)(action="search", query="design", folder="governance")
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

        result = await _manage(notes_dir)(
            action="update", filename="note.md", content="New content."
        )
        assert result["success"] is True

        text = (notes_dir / "note.md").read_text()
        assert "New content" in text
        assert "Original" in text  # title preserved
        assert "old" in text  # tags preserved

    async def test_update_tags(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: [old]\n---\n\nBody.\n'
        )

        result = await _manage(notes_dir)(
            action="update", filename="note.md", tags=["new", "updated"]
        )
        assert result["success"] is True

        text = (notes_dir / "note.md").read_text()
        assert "new" in text
        assert "Body." in text  # content preserved

    async def test_update_not_found(self, notes_dir):
        result = await _manage(notes_dir)(action="update", filename="nope.md", content="x")
        assert result["success"] is False

    async def test_update_nothing_to_update(self, notes_dir):
        (notes_dir / "note.md").write_text(
            '---\ntitle: "Note"\ndate: 2026-02-19\ntags: []\n---\n\nBody.\n'
        )
        result = await _manage(notes_dir)(action="update", filename="note.md")
        assert result["success"] is False
        assert "Provide" in result["error"]

    async def test_update_note_in_subfolder(self, notes_dir):
        sub = notes_dir / "docs"
        sub.mkdir()
        (sub / "note.md").write_text(
            '---\ntitle: "Doc"\ndate: 2026-03-23\ntags: []\n---\n\nOld body.\n'
        )

        result = await _manage(notes_dir)(
            action="update", filename="docs/note.md", content="New body."
        )
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

        result = await _manage(notes_dir)(action="delete", filename="doomed.md")
        assert result["success"] is True
        assert result["deleted"] is True
        assert not path.exists()

    async def test_delete_not_found(self, notes_dir):
        result = await _manage(notes_dir)(action="delete", filename="nope.md")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_delete_invalid_filename(self, notes_dir):
        result = await _manage(notes_dir)(action="delete", filename="../bad.md")
        assert result["success"] is False

    async def test_delete_note_in_subfolder(self, notes_dir):
        sub = notes_dir / "docs"
        sub.mkdir()
        path = sub / "doomed.md"
        path.write_text("content")

        result = await _manage(notes_dir)(action="delete", filename="docs/doomed.md")
        assert result["success"] is True
        assert result["deleted"] is True
        assert not path.exists()


# ---------------------------------------------------------------------------
# TestManageNotes — create_folder
# ---------------------------------------------------------------------------


class TestCreateFolder:
    async def test_create_folder(self, notes_dir):
        result = await _manage(notes_dir)(action="create_folder", folder="governance/tracks")
        assert result["success"] is True
        assert result["created"] is True
        assert (notes_dir / "governance" / "tracks").is_dir()

    async def test_create_folder_already_exists(self, notes_dir):
        (notes_dir / "existing").mkdir()
        result = await _manage(notes_dir)(action="create_folder", folder="existing")
        assert result["success"] is True
        assert result["created"] is False
        assert "already exists" in result["message"].lower()
        assert (await _manage(notes_dir)(action="create_folder", folder="."))["created"] is False

    async def test_create_folder_no_folder(self, notes_dir):
        result = await _manage(notes_dir)(action="create_folder")
        assert result["success"] is False
        assert "required" in result["error"].lower()

    async def test_create_folder_traversal(self, notes_dir):
        result = await _manage(notes_dir)(action="create_folder", folder="../outside")
        assert result["success"] is False
        assert "traversal" in result["error"].lower()

    async def test_create_nested_folder(self, notes_dir):
        result = await _manage(notes_dir)(action="create_folder", folder="a/b/c/d")
        assert result["success"] is True
        assert (notes_dir / "a" / "b" / "c" / "d").is_dir()


# ---------------------------------------------------------------------------
# TestManageNotes — move_note
# ---------------------------------------------------------------------------


class TestMoveNote:
    async def test_move_note(self, notes_dir):
        (notes_dir / "note.md").write_text("content")
        (notes_dir / "target").mkdir()

        result = await _manage(notes_dir)(action="move_note", filename="note.md", folder="target")
        assert result["success"] is True
        assert not (notes_dir / "note.md").exists()
        assert (notes_dir / "target" / "note.md").exists()
        assert result["to"] == "target/note.md"

    async def test_move_creates_destination(self, notes_dir):
        (notes_dir / "note.md").write_text("content")

        result = await _manage(notes_dir)(
            action="move_note", filename="note.md", folder="new/sub/dir"
        )
        assert result["success"] is True
        assert (notes_dir / "new" / "sub" / "dir" / "note.md").exists()

    async def test_move_no_filename(self, notes_dir):
        result = await _manage(notes_dir)(action="move_note", folder="target")
        assert result["success"] is False
        assert "filename" in result["error"].lower()

    async def test_move_no_folder(self, notes_dir):
        result = await _manage(notes_dir)(action="move_note", filename="note.md")
        assert result["success"] is False
        assert "folder" in result["error"].lower()

    async def test_move_source_not_found(self, notes_dir):
        result = await _manage(notes_dir)(action="move_note", filename="nope.md", folder="target")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_move_destination_conflict(self, notes_dir):
        (notes_dir / "note.md").write_text("original")
        target = notes_dir / "target"
        target.mkdir()
        (target / "note.md").write_text("existing")

        result = await _manage(notes_dir)(action="move_note", filename="note.md", folder="target")
        assert result["success"] is False
        assert "already exists" in result["error"].lower()

    async def test_move_traversal_on_folder(self, notes_dir):
        (notes_dir / "note.md").write_text("content")
        result = await _manage(notes_dir)(
            action="move_note", filename="note.md", folder="../outside"
        )
        assert result["success"] is False

    async def test_move_from_subfolder(self, notes_dir):
        """Can move a note from a subfolder to another subfolder."""
        src = notes_dir / "old"
        src.mkdir()
        (src / "note.md").write_text("content")

        result = await _manage(notes_dir)(action="move_note", filename="old/note.md", folder="new")
        assert result["success"] is True
        assert not (src / "note.md").exists()
        assert (notes_dir / "new" / "note.md").exists()


# ---------------------------------------------------------------------------
# TestManageNotes — invalid action
# ---------------------------------------------------------------------------


class TestInvalidAction:
    async def test_unknown_action(self, notes_dir):
        result = await _manage(notes_dir)(action="explode")
        assert result["success"] is False
        assert "Unknown action" in result["error"]


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegisterNotesTools:
    def test_tool_registered(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry, MarkdownNotes(tmp_path))
        assert "manage_notes" in registry.get_tool_names()

    def test_is_backend(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry, MarkdownNotes(tmp_path))
        assert registry._backend_definitions["manage_notes"].category == "backend"

    def test_has_handler(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry, MarkdownNotes(tmp_path))
        assert callable(registry._backend_handlers["manage_notes"])

    def test_schema_has_action_enum(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry, MarkdownNotes(tmp_path))
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

    def test_schema_has_folder_property(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        register_notes_tools(registry, MarkdownNotes(tmp_path))
        schema = registry._backend_definitions["manage_notes"].parameters_schema
        assert "folder" in schema["properties"]


class TestRobustness:
    def test_sequence_frontmatter_is_ignored(self, notes_dir):
        path = notes_dir / "odd.md"
        path.write_text("---\n- a\n- b\n---\nbody\n")
        parsed = _store(notes_dir).parse_note(path)
        assert parsed["title"] == "odd"
        assert parsed["content"] == "body"

    async def test_limit_zero_returns_nothing(self, notes_dir):
        await _manage(notes_dir)(action="create", title="One", content="alpha")
        listed = await _manage(notes_dir)(action="list", limit=0)
        assert listed["notes"] == []
        found = await _manage(notes_dir)(action="search", query="alpha", limit=0)
        assert found["results"] == []

    async def test_same_title_never_overwrites(self, notes_dir):
        import asyncio

        manage = _manage(notes_dir)
        results = await asyncio.gather(
            *(manage(action="create", title="Same", content=f"body {i}") for i in range(5))
        )
        names = {r["filename"] for r in results}
        assert len(names) == 5
        assert len(list(notes_dir.glob("*.md"))) == 5


@pytest.mark.parametrize("action", ["read", "update", "list", "search"])
async def test_notes_do_not_follow_symlinks_outside_root(tmp_path, action):
    root = tmp_path / "notes"
    root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside marker")
    (root / "linked.md").symlink_to(outside)
    result = await _manage(root)(
        action=action, filename="linked.md", content="changed", query="marker"
    )
    assert outside.read_text() == "outside marker"
    if action in ("read", "update"):
        assert result["success"] is False
    else:
        assert result["count"] == 0


async def test_notes_keep_in_root_symlinks_and_configured_symlink_root(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "original.md").write_text("inside marker")
    (root / "linked.md").symlink_to(root / "original.md")
    configured = tmp_path / "configured"
    configured.symlink_to(root, target_is_directory=True)
    assert (await _manage(configured)(action="read", filename="linked.md"))[
        "content"
    ] == "inside marker"
    result = await _manage(configured)(action="update", filename="linked.md", content="changed")
    assert result["success"] is True
    assert "changed" in (root / "original.md").read_text()


@pytest.mark.parametrize(
    ("action", "replace_parent"), [("read", False), ("update", False), ("read", True)]
)
async def test_notes_reject_entry_replaced_after_resolution(
    tmp_path, monkeypatch, action, replace_parent
):
    root = tmp_path / "notes"
    parent = root / "folder"
    parent.mkdir(parents=True)
    note = parent / "note.md"
    note.write_text("inside")
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir()
    outside = outside_parent / "note.md"
    outside.write_text("outside")
    original_open = os.open

    def replace_before_open(path, flags, *args, **kwargs):
        if replace_parent and path == "folder":
            parent.rename(root / "original")
            parent.symlink_to(outside_parent, target_is_directory=True)
        elif not replace_parent and path == "note.md":
            note.unlink()
            note.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_open)
    result = await _manage(root)(action=action, filename="folder/note.md", content="changed")
    assert result["success"] is False
    assert outside.read_text() == "outside"


async def test_note_create_uses_opened_parent_after_replacement(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    parent = root / "folder"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = os.open

    def replace_before_create(path, flags, *args, **kwargs):
        if flags & os.O_EXCL:
            parent.rename(root / "original")
            parent.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replace_before_create)
    result = await _manage(root)(action="create", title="New", content="body", folder="folder")
    assert result["success"] is True
    assert "body" in (root / "original" / result["filename"]).read_text()
    assert list(outside.iterdir()) == []


async def test_note_move_uses_both_opened_parents_after_replacement(tmp_path, monkeypatch):
    root = tmp_path / "notes"
    source, destination = root / "source", root / "destination"
    source.mkdir(parents=True)
    destination.mkdir()
    (source / "note.md").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.md").write_text("outside")
    original_rename = os.rename

    def replace_before_rename(src, dst, **kwargs):
        if "src_dir_fd" in kwargs:
            for directory in (source, destination):
                directory.rename(directory.with_name(directory.name + "-original"))
                directory.symlink_to(outside, target_is_directory=True)
        return original_rename(src, dst, **kwargs)

    monkeypatch.setattr(os, "rename", replace_before_rename)
    result = await _manage(root)(
        action="move_note", filename="source/note.md", folder="destination"
    )
    assert result["success"] is True
    assert (root / "destination-original" / "note.md").read_text() == "inside"
    assert (outside / "note.md").read_text() == "outside"


async def test_note_delete_removes_symlink_entry_not_its_target(tmp_path):
    target = tmp_path / "original.md"
    target.write_text("inside")
    (tmp_path / "linked.md").symlink_to("original.md")
    result = await _manage(tmp_path)(action="delete", filename="linked.md")
    assert result["success"] is True
    assert target.read_text() == "inside"
    assert not (tmp_path / "linked.md").is_symlink()


@pytest.mark.parametrize("cross_device", [False, True])
async def test_note_move_preserves_symlink_entry(tmp_path, monkeypatch, cross_device):
    target = tmp_path / "original.md"
    target.write_text("inside")
    (tmp_path / "linked.md").symlink_to("original.md")
    if cross_device:

        def fail_rename(*args, **kwargs):
            raise OSError(errno.EXDEV, "fixture")

        monkeypatch.setattr(os, "rename", fail_rename)
    result = await _manage(tmp_path)(action="move_note", filename="linked.md", folder="moved")
    assert result["success"] is True
    assert os.readlink(tmp_path / "moved" / "linked.md") == "original.md"
    assert target.read_text() == "inside"
    assert not (tmp_path / "linked.md").is_symlink()


async def test_note_move_across_devices_preserves_content_and_metadata(tmp_path, monkeypatch):
    source = tmp_path / "note.md"
    source.write_text("inside")
    source.chmod(0o640)
    os.utime(source, ns=(1_500_000_000_000_000_000, 1_500_000_001_000_000_000))
    metadata = source.stat()
    if hasattr(os, "setxattr"):
        os.setxattr(source, "user.runtime-test", b"fixture")

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "fixture")

    monkeypatch.setattr(os, "rename", cross_device)
    result = await _manage(tmp_path)(action="move_note", filename="note.md", folder="moved")
    assert result["success"] is True
    destination = tmp_path / "moved" / "note.md"
    copied = destination.stat()
    assert stat.S_IMODE(copied.st_mode) == stat.S_IMODE(metadata.st_mode)
    assert (copied.st_atime_ns, copied.st_mtime_ns) == (metadata.st_atime_ns, metadata.st_mtime_ns)
    assert destination.read_text() == "inside"
    assert not source.exists()
    if hasattr(os, "getxattr"):
        assert os.getxattr(destination, "user.runtime-test") == b"fixture"


async def test_failed_cross_device_copy_leaves_no_partial_destination(tmp_path, monkeypatch):
    (tmp_path / "note.md").write_text("inside")
    real_copy = shutil.copyfileobj

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "fixture")

    def full_disk(reader, writer):
        writer.write(b"part")
        raise OSError(errno.ENOSPC, "fixture")

    monkeypatch.setattr(os, "rename", cross_device)
    monkeypatch.setattr(shutil, "copyfileobj", full_disk)
    manage = _manage(tmp_path)
    with pytest.raises(OSError, match="fixture"):
        await manage(action="move_note", filename="note.md", folder="moved")
    assert not (tmp_path / "moved" / "note.md").exists()
    assert (tmp_path / "note.md").read_text() == "inside"

    monkeypatch.setattr(shutil, "copyfileobj", real_copy)
    result = await manage(action="move_note", filename="note.md", folder="moved")
    assert result["success"] is True
    assert (tmp_path / "moved" / "note.md").read_text() == "inside"


async def test_concurrent_moves_do_not_overwrite_a_note(tmp_path, monkeypatch):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "note.md").write_text(folder)
    real_rename = os.rename

    def slow_rename(*args, **kwargs):
        # Widen the gap between the destination check and the rename.
        time.sleep(0.1)
        return real_rename(*args, **kwargs)

    monkeypatch.setattr(os, "rename", slow_rename)
    notes = MarkdownNotes(tmp_path)
    results = await asyncio.gather(
        notes.move(filename="a/note.md", folder="target"),
        notes.move(filename="b/note.md", folder="target"),
    )
    assert sorted(r["success"] for r in results) == [False, True]
    assert "already exists" in next(r for r in results if not r["success"])["error"]
    remaining = [(tmp_path / f / "note.md") for f in ("a", "b")]
    kept = [p.read_text() for p in remaining if p.exists()]
    assert sorted(kept + [(tmp_path / "target" / "note.md").read_text()]) == ["a", "b"]


async def test_a_note_created_during_a_move_is_not_overwritten(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    notes = MarkdownNotes(tmp_path)
    created = await notes.create(title="Plan", content="moved", tags=None, folder="a")
    real_rename = os.rename
    renaming = threading.Event()

    def slow_rename(*args, **kwargs):
        # The move has checked its destination; hold the gap open.
        renaming.set()
        time.sleep(0.1)
        return real_rename(*args, **kwargs)

    monkeypatch.setattr(os, "rename", slow_rename)
    move = asyncio.create_task(notes.move(filename=created["path"], folder="b"))
    assert await asyncio.to_thread(renaming.wait, 5)
    made = await notes.create(title="Plan", content="new", tags=None, folder="b")
    moved = await move
    assert moved["success"] is True
    assert made["success"] is True
    contents = sorted(p.read_text().split("---")[-1].strip() for p in (tmp_path / "b").glob("*.md"))
    assert contents == ["moved", "new"]


@pytest.mark.skipif(not hasattr(os, "chflags"), reason="BSD file flags require macOS")
async def test_cross_device_move_with_file_flags_keeps_source(tmp_path, monkeypatch):
    source = tmp_path / "note.md"
    source.write_text("inside")
    os.chflags(source, stat.UF_NODUMP)

    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "fixture")

    monkeypatch.setattr(os, "rename", cross_device)
    try:
        with pytest.raises(OSError, match="cannot preserve file flags") as error:
            await _manage(tmp_path)(action="move_note", filename="note.md", folder="moved")
        assert error.value.errno == errno.ENOTSUP
        assert source.read_text() == "inside"
        assert not (tmp_path / "moved" / "note.md").exists()
    finally:
        os.chflags(source, 0)


async def test_invalid_note_encoding_still_raises(tmp_path):
    (tmp_path / "note.md").write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        await _manage(tmp_path)(action="read", filename="note.md")


async def test_note_read_through_execute_only_ancestor(tmp_path):
    ancestor = tmp_path / "search-only"
    root = ancestor / "notes"
    root.mkdir(parents=True)
    (root / "note.md").write_text("inside")
    ancestor.chmod(0o111)
    try:
        result = await _manage(root)(action="read", filename="note.md")
        assert result["success"] is True
        assert result["content"] == "inside"
    finally:
        ancestor.chmod(0o700)


async def test_cancelled_note_read_worker_closes_its_descriptors(tmp_path, monkeypatch):
    (tmp_path / "note.md").write_text("inside")
    store = _store(tmp_path)
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    opened = set()
    original_open, original_dup, original_close, original_fdopen = (
        os.open,
        os.dup,
        os.close,
        os.fdopen,
    )
    original_parse = store.parse_note

    def track_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.add(fd)
        return fd

    def track_dup(fd):
        duplicate = original_dup(fd)
        opened.add(duplicate)
        return duplicate

    def track_close(fd):
        opened.remove(fd)
        original_close(fd)

    def wait_before_read(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original_fdopen(*args, **kwargs)

    def parse(path):
        try:
            return original_parse(path)
        finally:
            finished.set()

    monkeypatch.setattr(os, "open", track_open)
    monkeypatch.setattr(os, "dup", track_dup)
    monkeypatch.setattr(os, "close", track_close)
    monkeypatch.setattr(os, "fdopen", wait_before_read)
    monkeypatch.setattr(store, "parse_note", parse)
    task = asyncio.create_task(store.read(filename="note.md"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert opened
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    assert not opened
