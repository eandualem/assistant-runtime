"""The version lifecycle: proposals, decisions, records, seeding and published events."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.base.events import EventHub
from assistant_runtime.services.artifacts._store import InMemoryArtifactStore
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactPermissionError,
    ArtifactVersionNotFoundError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor

HOST = Actor("host", "owner")
ASSISTANT = Actor("assistant", session_id="s-1")

PROFILE = AssistantProfile(
    name="shop",
    artifacts=(
        ArtifactDefinition(name="instructions", role="purpose", required=True, default="Help"),
        ArtifactDefinition(name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")),
    ),
)


async def _service(*, database=None) -> tuple[ArtifactService, list[dict]]:
    hub, seen = EventHub(), []
    hub.subscribe(seen.append)
    service = ArtifactService(ArtifactsConfig(), PROFILE, database_service=database, events=hub)
    await service.start()
    return service, seen


class TestProposals:
    async def test_a_proposal_is_pending_with_its_rationale_and_is_announced(self):
        service, seen = await _service()
        result = await service.propose(
            "instructions", "Help more", actor=ASSISTANT, rationale="Be warmer"
        )
        assert result.version.status == "pending"
        assert result.version.rationale == "Be warmer"
        assert result.version.actor_kind == "assistant"
        assert (await service.active_texts())["instructions"] == "Help"
        assert seen == [
            {
                "type": "artifact_proposal",
                "profile": "shop",
                "subject": None,
                "name": "instructions",
                "version": 1,
                "active_version": None,
                "rationale": "Be warmer",
                "proposed_by": {"kind": "assistant", "label": "assistant"},
                "session_id": "s-1",
            }
        ]

    async def test_approval_activates_supersedes_and_records_the_decision(self):
        service, seen = await _service()
        await service.update("instructions", "v1", actor=HOST)
        await service.propose("instructions", "v2", actor=ASSISTANT)
        seen.clear()
        result = await service.activate("instructions", 2, actor=HOST)
        assert result.version.status == "active"
        assert result.version.decided_by == "owner"
        assert (await service.active_texts())["instructions"] == "v2"
        history = {row.version: row.status for row in await service.history("instructions")}
        assert history == {2: "active", 1: "superseded"}
        assert seen[0]["type"] == "artifact_decision"
        assert seen[0]["status"] == "active"
        assert seen[0]["decided_by"] == "owner"

    async def test_rollback_is_not_a_decision(self):
        service, seen = await _service()
        await service.update("instructions", "v1", actor=HOST)
        await service.update("instructions", "v2", actor=HOST)
        seen.clear()
        result = await service.activate("instructions", 1, actor=HOST)
        assert result.version.status == "active"
        assert result.version.decided_by is None
        assert seen == []

    async def test_rejection_leaves_the_active_version(self):
        service, seen = await _service()
        await service.update("instructions", "v1", actor=HOST)
        await service.propose("instructions", "v2", actor=ASSISTANT)
        seen.clear()
        result = await service.reject("instructions", 2, actor=HOST, reason="Too long")
        assert result.version.status == "rejected"
        assert result.version.decision_reason == "Too long"
        assert result.activated is False
        assert result.live_version == 1
        assert (await service.active_texts())["instructions"] == "v1"
        assert seen[0]["type"] == "artifact_decision"
        assert seen[0]["status"] == "rejected"
        assert seen[0]["decision_reason"] == "Too long"

    async def test_only_a_pending_version_can_be_rejected(self):
        service, _ = await _service()
        await service.update("instructions", "v1", actor=HOST)
        with pytest.raises(ArtifactConflictError, match="active, not pending"):
            await service.reject("instructions", 1, actor=HOST)
        with pytest.raises(ArtifactVersionNotFoundError):
            await service.reject("instructions", 9, actor=HOST)

    async def test_rejecting_needs_the_activation_permission(self):
        service, _ = await _service()
        await service.propose("instructions", "v1", actor=ASSISTANT)
        with pytest.raises(ArtifactPermissionError):
            await service.reject("instructions", 1, actor=ASSISTANT)


class TestRecords:
    async def test_pending_proposals_with_a_diff_against_the_default(self):
        service, _ = await _service()
        await service.propose("instructions", "Help\nMore", actor=ASSISTANT, rationale="Why")
        [record] = await service.proposals()
        assert record["name"] == "instructions"
        assert record["role"] == "purpose"
        assert record["status"] == "pending"
        assert record["proposed_by"] == {"kind": "assistant", "label": "assistant"}
        assert record["active_version"] is None
        assert record["active_content"] == "Help"
        assert record["diff"].splitlines() == [
            "--- instructions (default)",
            "+++ instructions (v1)",
            "@@ -1 +1,2 @@",
            " Help",
            "+More",
        ]

    async def test_version_record_against_the_active_version(self):
        service, _ = await _service()
        await service.update("instructions", "Old", actor=HOST)
        await service.propose("instructions", "New", actor=ASSISTANT)
        record = await service.version_record("instructions", 2)
        assert record["active_version"] == 1
        assert record["active_content"] == "Old"
        assert "-Old\n+New\n" in record["diff"]
        with pytest.raises(ArtifactVersionNotFoundError):
            await service.version_record("instructions", 7)

    async def test_decided_proposals_leave_the_pending_list(self):
        service, _ = await _service()
        await service.propose("instructions", "A", actor=ASSISTANT)
        await service.propose("instructions", "B", actor=ASSISTANT)
        await service.reject("instructions", 1, actor=HOST)
        assert [r["version"] for r in await service.proposals()] == [2]
        assert [r["version"] for r in await service.proposals("rejected")] == [1]


class _DurableMemory(InMemoryArtifactStore):
    durable = True


class TestSeeding:
    async def _start(self, store, *, required: bool) -> ArtifactService:
        database = SimpleNamespace(healthy=True, required=required)
        with patch(
            "assistant_runtime.services.artifacts.interface.DatabaseArtifactStore",
            return_value=store,
        ):
            service, _ = await _service(database=database)
        return service

    async def test_required_database_stores_defaults_as_version_one(self):
        store = _DurableMemory()
        service = await self._start(store, required=True)
        [row] = await service.history("instructions")
        assert (row.version, row.status, row.actor_kind, row.content) == (
            1,
            "active",
            "seed",
            "Help",
        )
        assert await service.history("scratchpad") == []  # no default, nothing to seed

    async def test_seeding_happens_once_and_never_over_a_stored_version(self):
        store = _DurableMemory()
        service = await self._start(store, required=True)
        await service.update("instructions", "Edited", actor=HOST)
        await self._start(store, required=True)
        assert [row.version for row in await service.history("instructions")] == [2, 1]

    async def test_optional_database_keeps_defaults_unstored(self):
        service = await self._start(_DurableMemory(), required=False)
        assert await service.history("instructions") == []
