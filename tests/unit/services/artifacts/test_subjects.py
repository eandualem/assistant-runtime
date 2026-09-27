"""Subject-scoped artifacts, included profiles, collections and version retention."""

from __future__ import annotations

import pytest

from assistant_runtime.artifacts import (
    ArtifactCollection,
    ArtifactDefinition,
    ArtifactPolicy,
    AssistantProfile,
)
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactSubjectRequiredError,
)
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor

HOST = Actor("host", "owner")
ASSISTANT = Actor("assistant")
AUTONOMOUS = ArtifactPolicy(assistant_edit="autonomous")

TRACKER = AssistantProfile(
    name="tracker",
    artifacts=(
        ArtifactDefinition(name="instructions", required=True, default="Follow the agent"),
        ArtifactDefinition(
            name="progress", scope="subject", default="Nothing yet", policy=AUTONOMOUS
        ),
        ArtifactDefinition(name="notes", scope="subject", policy=AUTONOMOUS, keep_versions=2),
    ),
    include=("owner",),
)
OWNER = AssistantProfile(
    name="owner",
    artifacts=(
        ArtifactDefinition(name="preferences", default="Short answers"),
        ArtifactDefinition(name="private", scope="subject", default="not shared"),
    ),
)
LEAD = AssistantProfile(
    name="lead",
    artifacts=(ArtifactDefinition(name="instructions", required=True, default="Lead"),),
    collections=(ArtifactCollection(prefix="doc_", role="kept for the owner", policy=AUTONOMOUS),),
)


async def _service() -> ArtifactService:
    service = ArtifactService(ArtifactsConfig(), TRACKER, profiles=[OWNER, LEAD])
    await service.start()
    return service


class TestSubjects:
    async def test_each_subject_keeps_its_own_text(self):
        service = await _service()
        alpha, beta = service.for_subject("agent-a"), service.for_subject("agent-b")
        await alpha.update("progress", "Shipped the parser", actor=ASSISTANT)
        assert (await alpha.active_texts())["progress"] == "Shipped the parser"
        assert (await beta.active_texts())["progress"] == "Nothing yet"
        # Profile-scoped text is shared by every subject.
        await service.update("instructions", "Follow closely", actor=HOST)
        assert (await beta.active_texts())["instructions"] == "Follow closely"
        assert await service.subjects() == ["agent-a"]

    async def test_a_profile_view_leaves_subject_text_out(self):
        service = await _service()
        await service.for_subject("agent-a").update("progress", "Busy", actor=ASSISTANT)
        texts = await service.active_texts()
        assert texts["progress"] == ""
        assert texts["instructions"] == "Follow the agent"
        assert [row.name for row in await service.list_active()] == []

    async def test_subject_scoped_access_needs_a_subject(self):
        service = await _service()
        with pytest.raises(ArtifactSubjectRequiredError):
            await service.update("progress", "x", actor=ASSISTANT)
        with pytest.raises(ArtifactSubjectRequiredError):
            await service.history("progress")

    async def test_the_subject_view_cache_is_bounded(self):
        from assistant_runtime.services.artifacts import interface

        service = await _service()
        first = service.for_subject("agent-0")
        for n in range(1, interface._SUBJECT_VIEWS + 5):
            service.for_subject(f"agent-{n}")
        assert len(service._subject_views) == interface._SUBJECT_VIEWS
        assert "agent-0" not in service._subject_views
        assert service.for_subject("agent-0") is not first  # rebuilt on demand

    async def test_views_are_stable_per_subject(self):
        service = await _service()
        assert service.for_subject("agent-a") is service.for_subject("agent-a")
        assert service.for_subject(None) is service
        assert service.for_subject("agent-a").for_subject(None) is service

    async def test_proposal_events_and_records_name_the_subject(self):
        from assistant_runtime.base.events import EventHub

        hub, seen = EventHub(), []
        hub.subscribe(seen.append)
        profile = AssistantProfile(
            name="tracker",
            artifacts=(ArtifactDefinition(name="plan", scope="subject", default="None"),),
        )
        service = ArtifactService(ArtifactsConfig(), profile, events=hub)
        await service.start()
        await service.for_subject("agent-a").propose("plan", "Ship it", actor=ASSISTANT)
        assert seen[0]["subject"] == "agent-a"
        [record] = await service.proposals()
        assert record["subject"] == "agent-a"
        assert record["active_content"] == "None"


class TestStaleWrites:
    async def test_a_stale_write_returns_the_current_text(self):
        service = await _service()
        view = service.for_subject("agent-a")
        await view.update("progress", "First", actor=ASSISTANT)
        with pytest.raises(ArtifactConflictError) as caught:
            await view.update("progress", "Mine", actor=ASSISTANT, expected_version=0)
        assert caught.value.current_version == 1
        assert caught.value.current_content == "First"


class TestIncludes:
    async def test_included_profile_text_follows_in_the_prompt(self):
        service = await _service()
        await service.for_profile("owner").update("preferences", "Bullet points", actor=HOST)
        extras = await service.for_subject("agent-a").prompt_extras()
        assert extras == [("owner.preferences", "Bullet points")]

    async def test_including_an_unregistered_profile_fails_at_startup(self):
        with pytest.raises(ValueError, match="includes 'owner'"):
            ArtifactService(ArtifactsConfig(), TRACKER)


class TestCollections:
    async def test_the_assistant_creates_documents_listed_by_name(self):
        service = await _service()
        lead = service.for_profile("lead")
        await lead.update("doc_travel", "Flights on Friday", actor=ASSISTANT)
        assert (await lead.get_active("doc_travel")).content == "Flights on Friday"
        assert await lead.prompt_extras() == [
            (
                "documents",
                "Documents you keep (read one with manage_artifacts view): doc_travel.",
            )
        ]
        assert "doc_travel" not in await lead.active_texts()
        assert [row.name for row in await lead.list_active()] == ["doc_travel"]


class TestRetention:
    async def test_superseded_versions_beyond_the_limit_are_dropped(self):
        service = await _service()
        view = service.for_subject("agent-a")
        for text in ("one", "two", "three", "four", "five"):
            await view.update("notes", text, actor=ASSISTANT)
        history = [(row.version, row.status) for row in await view.history("notes")]
        assert history == [(5, "active"), (4, "superseded"), (3, "superseded")]

    async def test_versions_without_a_limit_are_all_kept(self):
        service = await _service()
        view = service.for_subject("agent-a")
        for text in ("one", "two", "three", "four"):
            await view.update("progress", text, actor=ASSISTANT)
        assert len(await view.history("progress")) == 4

    async def test_a_failing_document_listing_is_left_out_unless_the_database_is_required(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        from assistant_runtime.services.artifacts.exceptions import ArtifactError

        service = await _service()
        lead = service.for_profile("lead")
        service._store.get_all_active = AsyncMock(side_effect=RuntimeError("db gone"))
        assert await lead.prompt_extras() == []
        service._database_service = SimpleNamespace(required=True)
        lead._database_service = service._database_service
        with pytest.raises(ArtifactError, match="db gone"):
            await lead.prompt_extras()


class TestPromptVersions:
    async def test_versions_name_what_each_prompt_text_came_from(self):
        service = await _service()
        view = service.for_subject("agent-a")
        await view.update("progress", "Busy", actor=ASSISTANT)
        await service.for_profile("owner").update("preferences", "Bullets", actor=HOST)
        await service.for_profile("owner").update("preferences", "Lists", actor=HOST)
        assert await view.prompt_versions() == {
            "instructions": None,
            "progress": 1,
            "notes": None,
            "owner.preferences": 2,
        }
