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
    ArtifactTooLargeError,
    ArtifactVersionNotFoundError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor

HOST = Actor("host", "owner")
ASSISTANT = Actor("assistant", session_id="s-1")

PROFILE = AssistantProfile(
    name="shop",
    artifacts=(
        ArtifactDefinition(
            name="instructions",
            role="purpose",
            required=True,
            default="Help",
            policy=ArtifactPolicy(assistant_edit="propose"),
        ),
        ArtifactDefinition(name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")),
    ),
)


def _seeded(default: str) -> AssistantProfile:
    """The same profile with another default for its instructions."""
    instructions = ArtifactDefinition(
        name="instructions",
        role="purpose",
        required=True,
        default=default,
        policy=ArtifactPolicy(assistant_edit="propose"),
    )
    return AssistantProfile(name="shop", artifacts=(instructions, PROFILE.artifacts[1]))


async def _service(*, database=None, profile=PROFILE) -> tuple[ArtifactService, list[dict]]:
    hub, seen = EventHub(), []
    hub.subscribe(seen.append)
    service = ArtifactService(ArtifactsConfig(), profile, database_service=database, events=hub)
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

    async def test_a_rejected_version_is_not_activated_later(self):
        """A decision stands: reconsidering is a new proposal with its own record."""
        service, seen = await _service()
        await service.update("instructions", "v1", actor=HOST)
        await service.propose("instructions", "v2", actor=ASSISTANT)
        await service.reject("instructions", 2, actor=HOST, reason="No")
        seen.clear()
        with pytest.raises(ArtifactConflictError, match="was rejected"):
            await service.activate("instructions", 2, actor=HOST)
        assert (await service.active_texts())["instructions"] == "v1"
        [row] = [r for r in await service.history("instructions") if r.version == 2]
        assert (row.status, row.decision_reason) == ("rejected", "No")
        assert seen == []

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

    async def test_every_pending_proposal_is_listed_unless_limited(self):
        service, _ = await _service()
        for n in range(25):  # more than the history display limit
            await service.propose("instructions", f"v{n}", actor=ASSISTANT)
        assert len(await service.proposals()) == 25
        assert [r["version"] for r in await service.proposals(limit=2)] == [25, 24]

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
    async def _start(self, store, *, required: bool = True, profile=PROFILE) -> ArtifactService:
        database = SimpleNamespace(healthy=True, required=required)
        with patch(
            "assistant_runtime.services.artifacts.interface.DatabaseArtifactStore",
            return_value=store,
        ):
            service, _ = await _service(database=database, profile=profile)
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

    async def test_a_changed_default_replaces_a_seed_no_one_changed(self):
        store = _DurableMemory()
        await self._start(store)
        service = await self._start(store, profile=_seeded("Help more"))
        await self._start(store, profile=_seeded("Help more"))  # the same default: nothing new
        rows = await service.history("instructions")
        assert [(r.version, r.actor_kind, r.status) for r in rows] == [
            (2, "seed", "active"),
            (1, "seed", "superseded"),
        ]
        assert (await service.active_texts())["instructions"] == "Help more"

    async def test_a_host_or_assistant_version_stops_seed_refresh_for_good(self):
        edited, proposed = _DurableMemory(), _DurableMemory()
        service = await self._start(edited)
        await service.update("instructions", "Edited", actor=HOST)
        await service.activate("instructions", 1, actor=HOST)  # back to the seed
        service = await self._start(proposed)
        await service.propose("instructions", "Idea", actor=ASSISTANT)
        for store in (edited, proposed):
            service = await self._start(store, profile=_seeded("Help more"))
            assert [r.version for r in await service.history("instructions")] == [2, 1]
            assert (await service.active_texts())["instructions"] == "Help"

    async def test_an_emptied_default_removes_seeds_but_not_a_host_version(self):
        instructions = ArtifactDefinition(name="instructions", role="purpose")
        emptied = AssistantProfile(name="shop", artifacts=(instructions, PROFILE.artifacts[1]))
        seeded, edited = _DurableMemory(), _DurableMemory()
        await self._start(seeded)
        service = await self._start(seeded, profile=emptied)
        assert await service.history("instructions") == []
        assert not (await service.active_texts()).get("instructions")
        service = await self._start(edited)
        await service.update("instructions", "Edited", actor=HOST)
        service = await self._start(edited, profile=emptied)
        assert (await service.active_texts())["instructions"] == "Edited"

    async def test_a_rollback_keeps_the_seed_the_host_chose(self):
        store = _DurableMemory()
        await self._start(store)
        service = await self._start(store, profile=_seeded("Help more"))
        await service.activate("instructions", 1, actor=HOST)
        service = await self._start(store, profile=_seeded("Help most"))
        assert [r.version for r in await service.history("instructions")] == [2, 1]
        assert (await service.active_texts())["instructions"] == "Help"

    async def test_a_refresh_keeps_only_keep_versions_superseded_seeds(self):
        def kept(default: str) -> AssistantProfile:
            instructions = ArtifactDefinition(name="instructions", default=default, keep_versions=1)
            return AssistantProfile(name="shop", artifacts=(instructions,))

        store = _DurableMemory()
        for default in ("One", "Two", "Three", "Four"):
            service = await self._start(store, profile=kept(default))
        rows = await service.history("instructions")
        assert [(r.version, r.status) for r in rows] == [(4, "active"), (3, "superseded")]

    async def test_optional_database_keeps_defaults_unstored(self):
        service = await self._start(_DurableMemory(), required=False)
        assert await service.history("instructions") == []


class TestSizeBound:
    async def test_a_write_longer_than_max_chars_is_refused(self):
        profile = AssistantProfile(
            name="shop",
            artifacts=(
                ArtifactDefinition(
                    name="scratchpad",
                    policy=ArtifactPolicy(assistant_edit="autonomous"),
                    max_chars=10,
                ),
            ),
        )
        service, _ = await _service(profile=profile)
        await service.update("scratchpad", "Ten chars.", actor=ASSISTANT)
        for write in (service.update, service.propose):
            with pytest.raises(ArtifactTooLargeError, match="at most 10 characters.*has 11"):
                await write("scratchpad", "Eleven char", actor=HOST)
        assert [r.version for r in await service.history("scratchpad")] == [1]
