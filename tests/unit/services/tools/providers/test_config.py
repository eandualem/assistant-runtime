"""Tests for the providers configuration and provider construction."""

from __future__ import annotations

from pathlib import Path

import pytest

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

    def test_blank_library_path_is_rejected(self):
        with pytest.raises(ValueError, match="needs both a name and a path"):
            ProvidersConfig.from_env({"LIBRARY_PATHS": "docs="})

    def test_backbone_and_state_dir(self):
        config = ProvidersConfig.from_env(
            {
                "BACKBONE_URL": "http://127.0.0.1:7120",
                "BACKBONE_INFRASTRUCTURE_SESSIONS": "gateway, ngrok,",
                "AGENT_STATE_DIR": "~/.claude/state",
            }
        )
        assert config.backbone_url == "http://127.0.0.1:7120"
        assert config.backbone_infrastructure_sessions == frozenset({"gateway", "ngrok"})
        assert config.agent_state_dir == Path("~/.claude/state").expanduser()
        assert config.configured() == ["backbone", "claude_code"]

    def test_github_and_telegram(self):
        config = ProvidersConfig.from_env(
            {
                "GITHUB_TOKEN": "t",
                "GITHUB_REPO_OWNER": "org",
                "GITHUB_REPO_NAME": "repo",
                "TELEGRAM_TOKEN": "bot",
                "TELEGRAM_CHAT_ID": "1",
            }
        )
        assert config.github_repo == "org/repo"
        assert config.configured() == ["github", "telegram"]
        assert ProvidersConfig.from_env({"GITHUB_TOKEN": "t"}).configured() == []
        assert set(build_providers(config)) == {"issues", "messaging"}

    def test_bare_library_path_is_the_default_collection(self):
        config = ProvidersConfig.from_env({"LIBRARY_PATHS": "/srv/docs"})
        assert config.library_paths == {"default": Path("/srv/docs")}


class TestBuildProviders:
    def test_only_configured_capabilities_get_a_provider(self, tmp_path):
        providers = build_providers(ProvidersConfig(notes_path=tmp_path))
        assert set(providers) == {"notes"}
        assert isinstance(providers["notes"], MarkdownNotes)

    def test_backbone_serves_six_capabilities(self):
        providers = build_providers(ProvidersConfig(backbone_url="http://x"))
        assert set(providers) == {
            "peers",
            "rooms",
            "reminders",
            "activity",
            "workgroups",
            "repositories",
        }

    def test_state_dir_serves_approvals(self, tmp_path):
        providers = build_providers(ProvidersConfig(agent_state_dir=tmp_path))
        assert set(providers) == {"approvals"}

    def test_library_provider(self, tmp_path):
        providers = build_providers(ProvidersConfig(library_paths={"g": tmp_path}))
        assert isinstance(providers["library"], FilesystemLibrary)
