"""The shipped documentation: discoverable from a checkout and printable from the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant_runtime import help as help_module
from assistant_runtime.cli import main
from assistant_runtime.help import docs_dir, get_doc, list_docs

EXPECTED_PAGES = {"api", "concepts", "configuration", "getting-started"}


class TestCheckout:
    def test_docs_dir_is_the_repository_docs(self):
        directory = docs_dir()
        assert directory is not None
        assert directory.name == "docs"
        assert (directory / "README.md").is_file()

    def test_pages_present_with_summaries(self):
        pages = {p["name"]: p["summary"] for p in list_docs()}
        assert set(pages) >= EXPECTED_PAGES
        assert "README" not in pages
        assert all(summary for summary in pages.values())

    def test_get_doc(self):
        assert get_doc("concepts").startswith("# Concepts")
        assert get_doc("nope") is None
        assert get_doc("../README") is None

    @pytest.mark.parametrize("name", sorted(EXPECTED_PAGES))
    def test_pages_have_no_private_names(self, name):
        text = get_doc(name).lower()
        for private in ("lovely", "jarvis", "loveble"):
            assert private not in text


class TestWithoutDocs:
    def test_nothing_shipped(self, monkeypatch, tmp_path: Path):
        monkeypatch.setattr(help_module, "_DOCS_DIRS", (tmp_path / "none",))
        assert docs_dir() is None
        assert list_docs() == []
        assert get_doc("concepts") is None


class TestCli:
    def test_docs_lists_pages(self, capsys):
        assert main(["docs"]) == 0
        out = capsys.readouterr().out
        for name in EXPECTED_PAGES:
            assert name in out

    def test_help_is_an_alias(self, capsys):
        assert main(["help", "concepts"]) == 0
        assert capsys.readouterr().out.startswith("# Concepts")

    def test_unknown_page(self, capsys):
        assert main(["docs", "nope"]) == 1
        assert "unknown page" in capsys.readouterr().out

    def test_nothing_shipped_points_to_github(self, monkeypatch, tmp_path: Path, capsys):
        monkeypatch.setattr(help_module, "_DOCS_DIRS", (tmp_path / "none",))
        assert main(["docs"]) == 1
        assert "github.com" in capsys.readouterr().out
