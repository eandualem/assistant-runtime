"""Tests for the providers configuration and provider construction."""

from __future__ import annotations

from pathlib import Path

from assistant_runtime.services.tools.providers import build_providers
from assistant_runtime.services.tools.providers.config import ProvidersConfig
from assistant_runtime.services.tools.providers.filesystem import (
    FilesystemLibrary,
    MarkdownNotes,
)


class TestProvidersConfig:
    def test_nothing_configured_by_default(self):
        config = ProvidersConfig.from_env({})
        assert config.notes_path is None
        assert config.library_paths == {}
        assert config.configured() == []

    def test_reads_conventional_variables(self):
        config = ProvidersConfig.from_env(
            {"NOTES_PATH": "~/notes", "LIBRARY_PATHS": "global=~/.claude/skills, team=/srv/docs"}
        )
        assert config.notes_path == Path("~/notes").expanduser()
        assert config.library_paths == {
            "global": Path("~/.claude/skills").expanduser(),
            "team": Path("/srv/docs"),
        }
        assert config.configured() == ["notes", "library"]

    def test_bare_library_path_is_the_default_collection(self):
        config = ProvidersConfig.from_env({"LIBRARY_PATHS": "/srv/docs"})
        assert config.library_paths == {"default": Path("/srv/docs")}


class TestBuildProviders:
    def test_only_configured_capabilities_get_a_provider(self, tmp_path):
        providers = build_providers(ProvidersConfig(notes_path=tmp_path))
        assert set(providers) == {"notes"}
        assert isinstance(providers["notes"], MarkdownNotes)

    def test_library_provider(self, tmp_path):
        providers = build_providers(ProvidersConfig(library_paths={"g": tmp_path}))
        assert isinstance(providers["library"], FilesystemLibrary)
