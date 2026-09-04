"""Bundled default artifacts and the database overlay in AssistantService."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.app.assistant._defaults import DEFAULTS_DIR, load_default_artifacts
from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.artifacts import REQUIRED_ARTIFACT_NAMES


class TestLoadDefaultArtifacts:
    def test_every_required_artifact_ships(self):
        artifacts = load_default_artifacts()
        assert set(artifacts) == set(REQUIRED_ARTIFACT_NAMES)
        assert all(text.strip() for text in artifacts.values())

    def test_files_live_inside_the_package(self):
        for name in REQUIRED_ARTIFACT_NAMES:
            assert (DEFAULTS_DIR / f"{name}.md").is_file()

    def test_defaults_carry_no_private_names(self):
        blob = "\n".join(load_default_artifacts().values()).lower()
        for private in ("lovely", "jarvis", "loveble"):
            assert private not in blob


def _service(database: Any) -> AssistantService:
    return AssistantService(
        config=AssistantConfig(),
        llm_service=MagicMock(),
        history_service=AsyncMock(),
        tool_service=MagicMock(),
        database_service=database,
    )


def _database(*, healthy: bool, rows: list[Any] | None = None, fail: bool = False) -> Any:
    @asynccontextmanager
    async def session_context():
        if fail:
            raise RuntimeError("boom")
        yield object()

    return SimpleNamespace(healthy=healthy, session_context=session_context, _rows=rows or [])


def _patch_repo(rows: list[Any]):
    repo = MagicMock()
    repo.get_all_active = AsyncMock(return_value=rows)
    return patch(
        "assistant_runtime.services.database.repositories.ArtifactRepository",
        return_value=repo,
    )


class TestArtifactLoading:
    async def test_no_database_uses_defaults(self):
        service = _service(None)
        assert await service._load_active_artifacts() == load_default_artifacts()

    async def test_unreachable_database_uses_defaults(self):
        service = _service(_database(healthy=False))
        assert await service._load_active_artifacts() == load_default_artifacts()

    async def test_database_rows_override_defaults(self):
        rows = [SimpleNamespace(name="persona", content="Custom persona")]
        service = _service(_database(healthy=True))
        with _patch_repo(rows):
            artifacts = await service._load_active_artifacts()
        assert artifacts["persona"] == "Custom persona"
        assert artifacts["soul"] == load_default_artifacts()["soul"]

    async def test_database_failure_falls_back_to_defaults(self):
        service = _service(_database(healthy=True, fail=True))
        assert await service._load_active_artifacts() == load_default_artifacts()

    async def test_overlay_is_cached(self):
        rows = [SimpleNamespace(name="ecosystem", content="Team A")]
        service = _service(_database(healthy=True))
        with _patch_repo(rows) as repo_cls:
            await service._load_active_artifacts()
            await service._load_active_artifacts()
        assert repo_cls.call_count == 1


class TestSessionPersistenceMode:
    async def test_unhealthy_database_means_memory_only_sessions(self):
        service = _service(_database(healthy=False))
        await service.start()
        assert service.get_session_store().persistent is False

    async def test_healthy_database_is_passed_to_the_store(self):
        db = _database(healthy=True)
        service = _service(db)
        await service.start()
        assert service.get_session_store().persistent is True


@pytest.mark.parametrize("name", REQUIRED_ARTIFACT_NAMES)
def test_default_text_is_stripped(name: str):
    text = load_default_artifacts()[name]
    assert text == text.strip()
