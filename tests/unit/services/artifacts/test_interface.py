"""ArtifactService: store selection, prompt texts, and policy enforcement."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.services.artifacts._store import (
    DatabaseArtifactStore,
    InMemoryArtifactStore,
)
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactError,
    ArtifactPermissionError,
    ArtifactVersionNotFoundError,
    UnknownArtifactError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor

ASSISTANT = Actor("assistant")
HOST = Actor("host", "operator")


def _profile(name: str = "default") -> AssistantProfile:
    return AssistantProfile(
        name=name,
        artifacts=(
            ArtifactDefinition(name="instructions", required=True, default="Default help"),
            ArtifactDefinition(
                name="persona",
                default="Warm",
                policy=ArtifactPolicy(assistant_edit="autonomous", assistant_activate=True),
            ),
            ArtifactDefinition(
                name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")
            ),
            ArtifactDefinition(
                name="policies",
                default="Refunds within 30 days",
                policy=ArtifactPolicy(assistant_edit="none", host_edit=False),
            ),
        ),
    )


async def _service(
    profile: AssistantProfile | None = None, *, database=None, ttl: float = 5.0
) -> ArtifactService:
    service = ArtifactService(
        ArtifactsConfig(cache_ttl_seconds=ttl), profile or _profile(), database_service=database
    )
    await service.start()
    return service


class TestLifecycle:
    async def test_memory_store_without_database(self):
        service = await _service()
        assert service.durable is False
        assert (await service.health_check()) == {
            "healthy": True,
            "profile": "default",
            "artifacts": 4,
            "durable": False,
        }

    async def test_memory_store_when_database_is_unreachable(self):
        service = await _service(database=SimpleNamespace(healthy=False))
        assert service.durable is False

    async def test_database_store_when_reachable(self):
        service = await _service(database=SimpleNamespace(healthy=True))
        assert isinstance(service._store, DatabaseArtifactStore)
        assert service.durable is True

    async def test_stopped_service_refuses_reads(self):
        service = await _service()
        await service.stop()
        with pytest.raises(ArtifactError, match="not started"):
            await service.active_texts()
        assert (await service.health_check())["healthy"] is False


class TestActiveTexts:
    async def test_defaults_until_a_version_is_active(self):
        service = await _service()
        assert await service.active_texts() == {
            "instructions": "Default help",
            "persona": "Warm",
            "scratchpad": "",
            "policies": "Refunds within 30 days",
        }

    async def test_active_version_overrides_default_on_the_next_read(self):
        service = await _service()
        await service.active_texts()
        await service.update("persona", "Direct", actor=ASSISTANT)
        assert (await service.active_texts())["persona"] == "Direct"

    async def test_proposed_version_does_not_change_the_prompt(self):
        service = await _service()
        await service.propose("instructions", "New help", actor=ASSISTANT)
        assert (await service.active_texts())["instructions"] == "Default help"

    async def test_reads_are_cached_until_invalidated_or_expired(self):
        service = await _service(ttl=3600)
        service._store.get_all_active = AsyncMock(return_value=[])
        await service.active_texts()
        await service.active_texts()
        assert service._store.get_all_active.await_count == 1
        service.invalidate()
        await service.active_texts()
        assert service._store.get_all_active.await_count == 2

    async def test_store_failure_falls_back_to_defaults(self):
        service = await _service()
        service._store.get_all_active = AsyncMock(side_effect=RuntimeError("down"))
        assert (await service.active_texts())["instructions"] == "Default help"

    async def test_rows_outside_the_profile_are_ignored(self):
        service = await _service()
        await service._store.propose("default", "legacy", "old text", "system")
        await service._store.activate("default", "legacy", 1)
        assert "legacy" not in await service.active_texts()
        assert [row.name for row in await service.list_active()] == []


class TestPolicies:
    async def test_assistant_proposal_stays_inactive(self):
        service = await _service()
        result = await service.propose("instructions", "New help", actor=ASSISTANT)
        assert result.activated is False
        assert result.live_version is None
        assert result.version.version == 1
        assert result.version.proposed_by == "assistant"
        assert result.durable is False

    async def test_assistant_cannot_activate_a_review_required_artifact(self):
        service = await _service()
        await service.propose("instructions", "New help", actor=ASSISTANT)
        with pytest.raises(ArtifactPermissionError, match="may not activate"):
            await service.activate("instructions", 1, actor=ASSISTANT)
        with pytest.raises(ArtifactPermissionError, match="may not update"):
            await service.update("instructions", "Sneaky", actor=ASSISTANT)

    async def test_host_activates_the_proposal(self):
        service = await _service()
        await service.propose("instructions", "New help", actor=ASSISTANT)
        result = await service.activate("instructions", 1, actor=HOST)
        assert result.activated
        assert result.live_version == 1
        assert (await service.active_texts())["instructions"] == "New help"

    async def test_autonomous_artifacts_activate_at_once(self):
        service = await _service()
        result = await service.update("scratchpad", "Remember X", actor=ASSISTANT)
        assert result.activated
        assert result.live_version == 1
        assert (await service.get_active("scratchpad")).content == "Remember X"

    async def test_explicitly_autonomous_persona_can_be_rolled_back_by_the_assistant(self):
        service = await _service()
        await service.update("persona", "v1", actor=ASSISTANT)
        await service.update("persona", "v2", actor=ASSISTANT)
        result = await service.activate("persona", 1, actor=ASSISTANT)
        assert result.live_version == 1
        assert (await service.active_texts())["persona"] == "v1"

    async def test_read_only_artifact_rejects_every_writer(self):
        service = await _service()
        for actor in (ASSISTANT, HOST):
            with pytest.raises(ArtifactPermissionError):
                await service.propose("policies", "x", actor=actor)
            with pytest.raises(ArtifactPermissionError):
                await service.update("policies", "x", actor=actor)

    async def test_assistant_never_deletes(self):
        service = await _service()
        with pytest.raises(ArtifactPermissionError, match="may not delete"):
            await service.delete("scratchpad", actor=ASSISTANT)

    async def test_host_delete_restores_the_default(self):
        service = await _service()
        await service.update("persona", "Direct", actor=HOST)
        assert await service.delete("persona", actor=HOST) == 1
        assert (await service.active_texts())["persona"] == "Warm"

    async def test_label_does_not_widen_permissions(self):
        service = await _service()
        with pytest.raises(ArtifactPermissionError):
            await service.update("instructions", "x", actor=Actor("assistant", "host"))

    async def test_allowed_actions_follow_the_policy(self):
        service = await _service()
        assert service.allowed_actions("instructions", "assistant") == ["propose"]
        assert service.allowed_actions("persona", "assistant") == ["propose", "update", "activate"]
        assert service.allowed_actions("policies", "assistant") == []
        assert service.allowed_actions("policies", "host") == []
        assert service.allowed_actions("instructions", "host") == [
            "propose",
            "update",
            "activate",
            "delete",
        ]


class TestVersions:
    async def test_unknown_names_are_rejected_everywhere(self):
        service = await _service()
        with pytest.raises(UnknownArtifactError, match="Known artifacts"):
            await service.get_active("soul")
        with pytest.raises(UnknownArtifactError):
            await service.propose("soul", "x", actor=HOST)
        with pytest.raises(UnknownArtifactError):
            service.allowed_actions("soul", "host")

    async def test_stale_expected_version_conflicts(self):
        service = await _service()
        await service.update("scratchpad", "v1", actor=ASSISTANT)
        with pytest.raises(ArtifactConflictError, match="at version 1, not 0"):
            await service.update("scratchpad", "v2", actor=ASSISTANT, expected_version=0)
        result = await service.update("scratchpad", "v2", actor=ASSISTANT, expected_version=1)
        assert result.live_version == 2

    async def test_duplicate_write_records_nothing(self):
        service = await _service()
        await service.update("scratchpad", "same", actor=ASSISTANT)
        result = await service.update("scratchpad", " same \n", actor=ASSISTANT)
        assert result.unchanged
        assert result.live_version == 1
        assert len(await service.history("scratchpad")) == 1

    async def test_empty_content_is_rejected(self):
        service = await _service()
        with pytest.raises(ArtifactError, match="Content is required"):
            await service.update("scratchpad", "   ", actor=ASSISTANT)

    async def test_activating_a_missing_version_fails(self):
        service = await _service()
        with pytest.raises(ArtifactVersionNotFoundError):
            await service.activate("persona", 9, actor=HOST)

    async def test_history_is_newest_first_and_limited(self):
        service = await _service()
        for text in ("a", "b", "c"):
            await service.update("scratchpad", text, actor=ASSISTANT)
        rows = await service.history("scratchpad", limit=2)
        assert [(r.version, r.is_active) for r in rows] == [(3, True), (2, False)]

    async def test_list_active_is_in_prompt_order(self):
        service = await _service()
        await service.update("scratchpad", "s", actor=ASSISTANT)
        await service.update("persona", "p", actor=ASSISTANT)
        assert [r.name for r in await service.list_active()] == ["persona", "scratchpad"]


class TestScoping:
    async def test_two_profiles_sharing_a_store_never_see_each_other(self):
        store = InMemoryArtifactStore()
        first = await _service(_profile("shop"))
        second = await _service(_profile("clinic"))
        first._store = second._store = store
        await first.update("persona", "Shop voice", actor=HOST)
        assert (await second.active_texts())["persona"] == "Warm"
        assert await second.get_active("persona") is None
        await second.update("persona", "Clinic voice", actor=HOST)
        assert (await first.active_texts())["persona"] == "Shop voice"
        assert await store.get_history("shop", "persona", 10) != await store.get_history(
            "clinic", "persona", 10
        )
