"""Tests for the providers configuration and provider construction."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
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
                "BACKBONE_INFRASTRUCTURE_SESSIONS": "service-a, service-b,",
                "AGENT_STATE_DIR": "~/.claude/state",
            }
        )
        assert config.backbone_url == "http://127.0.0.1:7120"
        assert config.backbone_infrastructure_sessions == frozenset({"service-a", "service-b"})
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

    @pytest.mark.parametrize("ambient", ["", "ambient"])
    async def test_instances_use_their_supplied_config_and_keep_backbone_caches_separate(
        self, monkeypatch, ambient
    ):
        requests = []

        async def request(client, method, url, **kwargs):
            sent = httpx.Request(
                method, url, headers=kwargs.get("headers"), json=kwargs.get("json")
            )
            requests.append(sent)
            body = {"items": [], "total": 0, "number": 1, "result": {"message_id": 1}}
            if sent.url.path == "/api/agents":
                body["items"] = [{"session": sent.url.host, "online": True, "state": "idle"}]
            return httpx.Response(200, json=body, request=sent)

        monkeypatch.setattr(httpx.AsyncClient, "request", request)
        configured = []
        for name, key in (("first", ""), ("second", "test-second-key")):
            # Backbone's existing API-key source remains environment-only, captured at build.
            monkeypatch.setenv("BACKBONE_API_KEY", key)
            config = ProvidersConfig(
                backbone_url=f"https://{name}.example",
                github_repo=f"{name}/repo",
                github_token=f"test-github-{name}",
                telegram_token=f"test-telegram-{name}",
                telegram_chat_id=f"chat-{name}",
            )
            configured.append((name, key, build_providers(config)))

        for variable in (
            "BACKBONE_URL",
            "BACKBONE_API_KEY",
            "GITHUB_TOKEN",
            "GITHUB_REPO_OWNER",
            "GITHUB_REPO_NAME",
            "TELEGRAM_TOKEN",
            "TELEGRAM_CHAT_ID",
        ):
            monkeypatch.setenv(variable, ambient)

        async def exercise(name, key, providers):
            results = await asyncio.gather(
                providers["issues"].create_issue("Test", "Body", []),
                providers["messaging"].respond_telegram("Hello"),
                providers["peers"].get_active_agents(),
                providers["rooms"].list_meeting_rooms(),
                providers["reminders"].add_schedule_item("12:00", "Test"),
                providers["activity"].get_delivery_status(),
                providers["workgroups"].list_swarms(),
                providers["repositories"].list_repos(),
            )
            assert all(result["success"] for result in results)
            assert results[1]["chat_id"] == f"chat-{name}"
            assert results[2]["agents"][0]["session_name"] == f"{name}.example"
            cached = await providers["peers"].get_active_agents()
            assert cached == results[2]

        await asyncio.gather(*(exercise(*item) for item in configured))
        for name, key, _ in configured:
            backbone = [request for request in requests if request.url.host == f"{name}.example"]
            assert (
                len(backbone) == 6
            )  # The repeated registry read was served by this instance's cache.
            assert all(
                request.headers.get("authorization") == (f"Bearer {key}" if key else None)
                for request in backbone
            )
            issue = next(
                request for request in requests if request.url.path == f"/repos/{name}/repo/issues"
            )
            assert issue.headers["authorization"] == f"Bearer test-github-{name}"
            assert any(
                request.url.path == f"/bottest-telegram-{name}/sendMessage" for request in requests
            )
        assert len(requests) == 16

    def test_missing_config_credentials_do_not_enable_ambient_providers(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "test-ambient-token")
        monkeypatch.setenv("TELEGRAM_TOKEN", "test-ambient-token")
        config = ProvidersConfig(github_repo="example/repo", telegram_chat_id="test-chat")
        assert build_providers(config) == {}
