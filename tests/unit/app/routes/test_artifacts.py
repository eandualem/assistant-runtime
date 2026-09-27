"""/artifacts routes against a real ArtifactService on an in-memory store."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.artifacts import proposals_router, router
from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor

PROFILE = AssistantProfile(
    name="shop",
    artifacts=(
        ArtifactDefinition(name="instructions", role="purpose", required=True, default="Help"),
        ArtifactDefinition(name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")),
        ArtifactDefinition(
            name="policies", default="Refunds", policy=ArtifactPolicy(host_edit=False)
        ),
    ),
)


@pytest.fixture
async def artifacts() -> ArtifactService:
    service = ArtifactService(ArtifactsConfig(), PROFILE)
    await service.start()
    return service


@pytest.fixture
async def client(artifacts):
    app = FastAPI()
    app.include_router(router)
    app.state.artifact_service = artifacts
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestReads:
    async def test_list_only_active_versions_in_prompt_order(self, client, artifacts):
        response = await client.get("/artifacts")
        assert response.status_code == 200
        assert response.json() == []
        await artifacts.update("scratchpad", "note", actor=Actor("host"))
        await artifacts.update("instructions", "Help more", actor=Actor("host"))
        names = [row["name"] for row in (await client.get("/artifacts")).json()]
        assert names == ["instructions", "scratchpad"]

    async def test_profile_describes_policies(self, client, artifacts):
        await artifacts.update("scratchpad", "note", actor=Actor("host"))
        body = (await client.get("/artifacts/profile")).json()
        assert body["name"] == "shop"
        assert body["durable"] is False
        assert [a["name"] for a in body["artifacts"]] == ["instructions", "scratchpad", "policies"]
        assert body["artifacts"][1]["live_version"] == 1
        assert body["artifacts"][2]["policy"] == {
            "assistant_edit": "propose",
            "assistant_activate": False,
            "host_edit": False,
        }

    async def test_get_default_then_stored(self, client, artifacts):
        body = (await client.get("/artifacts/instructions")).json()
        assert body["content"] == "Help"
        assert body["source"] == "default"
        assert body["version"] is None
        await artifacts.update("instructions", "Help more", actor=Actor("host"))
        body = (await client.get("/artifacts/instructions")).json()
        assert body["content"] == "Help more"
        assert body["source"] == "store"
        assert body["version"] == 1
        assert body["is_active"] is True

    async def test_unknown_name_is_422(self, client):
        response = await client.get("/artifacts/soul")
        assert response.status_code == 422
        assert "Known artifacts" in response.json()["detail"]

    async def test_history_with_limit(self, client, artifacts):
        for text in ("a", "b", "c"):
            await artifacts.update("scratchpad", text, actor=Actor("host"))
        response = await client.get("/artifacts/scratchpad/history", params={"limit": 2})
        assert [row["version"] for row in response.json()] == [3, 2]

    async def test_no_service_is_503(self):
        app = FastAPI()
        app.include_router(router)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/artifacts")).status_code == 503


class TestWrites:
    async def test_propose_then_approve(self, client, artifacts):
        response = await client.post(
            "/artifacts/instructions/propose", json={"content": "Help more"}
        )
        assert response.status_code == 201
        body = response.json()
        assert body["version"] == 1
        assert body["is_active"] is False
        assert body["effective_on_next_request"] is False
        assert body["live_version"] is None
        assert body["proposed_by"] == "local"  # the authenticated principal, not the body
        assert body["durable"] is False
        assert (await artifacts.active_texts())["instructions"] == "Help"

        response = await client.post("/artifacts/instructions/approve/1")
        assert response.status_code == 200
        assert response.json()["effective_on_next_request"] is True
        assert (await artifacts.active_texts())["instructions"] == "Help more"

    async def test_patch_updates_and_activates(self, client, artifacts):
        response = await client.patch("/artifacts/scratchpad", json={"content": "note"})
        assert response.status_code == 200
        assert response.json()["live_version"] == 1
        assert (await artifacts.active_texts())["scratchpad"] == "note"

    async def test_rollback(self, client, artifacts):
        for text in ("v1", "v2"):
            await artifacts.update("scratchpad", text, actor=Actor("host"))
        response = await client.post("/artifacts/scratchpad/rollback/1")
        assert response.status_code == 200
        assert "Rolled back" in response.json()["message"]
        assert (await artifacts.active_texts())["scratchpad"] == "v1"

    async def test_missing_version_is_404(self, client):
        assert (await client.post("/artifacts/scratchpad/approve/9")).status_code == 404

    async def test_policy_denial_is_403(self, client):
        response = await client.patch("/artifacts/policies", json={"content": "x"})
        assert response.status_code == 403
        assert "may not update" in response.json()["detail"]

    async def test_stale_expected_version_is_409(self, client, artifacts):
        await artifacts.update("scratchpad", "v1", actor=Actor("host"))
        response = await client.patch(
            "/artifacts/scratchpad", json={"content": "v2", "expected_version": 0}
        )
        assert response.status_code == 409

    async def test_duplicate_content_is_reported_unchanged(self, client, artifacts):
        await artifacts.update("scratchpad", "v1", actor=Actor("host"))
        response = await client.patch("/artifacts/scratchpad", json={"content": "v1"})
        assert response.json()["unchanged"] is True

    async def test_empty_content_is_422(self, client):
        assert (
            await client.patch("/artifacts/scratchpad", json={"content": ""})
        ).status_code == 422

    async def test_actions_endpoint(self, client, artifacts):
        propose = await client.post(
            "/artifacts/instructions/actions", json={"action": "propose", "content": "Help more"}
        )
        assert propose.status_code == 200
        assert propose.json()["version"] == 1
        approve = await client.post(
            "/artifacts/instructions/actions", json={"action": "approve", "version": 1}
        )
        assert approve.json()["live_version"] == 1
        update = await client.post(
            "/artifacts/scratchpad/actions", json={"action": "update", "content": "note"}
        )
        assert update.json()["effective_on_next_request"] is True
        rollback = await client.post("/artifacts/instructions/actions", json={"action": "rollback"})
        assert rollback.status_code == 422
        bad = await client.post("/artifacts/instructions/actions", json={"action": "propose"})
        assert bad.status_code == 422
        unknown = await client.post("/artifacts/instructions/actions", json={"action": "delete"})
        assert unknown.status_code == 422

    async def test_delete(self, client, artifacts):
        assert (await client.delete("/artifacts/scratchpad")).status_code == 404
        await artifacts.update("scratchpad", "v1", actor=Actor("host"))
        response = await client.delete("/artifacts/scratchpad")
        assert response.json() == {
            "success": True,
            "name": "scratchpad",
            "deleted_versions": 1,
            "durable": False,
        }
        assert (await client.delete("/artifacts/policies")).status_code == 403


class TestProfileSelection:
    @pytest.fixture
    async def artifacts(self):
        other = AssistantProfile(
            name="clinic",
            artifacts=(
                ArtifactDefinition(name="instructions", default="Clinic help"),
                ArtifactDefinition(name="policies", policy=ArtifactPolicy(host_edit=False)),
            ),
        )
        service = ArtifactService(ArtifactsConfig(), PROFILE, profiles=[other])
        await service.start()
        yield service
        await service.stop()

    async def test_discovery_and_mutations_are_scoped(self, client, artifacts):
        for params, selected in [({}, "shop"), ({"profile": "clinic"}, "clinic")]:
            response = await client.get("/artifacts/profile", params=params)
            assert response.status_code == 200
            assert response.json()["name"] == selected
            assert response.json()["available_profiles"] == ["shop", "clinic"]
        params = {"profile": "clinic"}
        response = await client.post(
            "/artifacts/instructions/propose", params=params, json={"content": "Clinic v1"}
        )
        assert response.status_code == 201
        assert (await client.get("/artifacts/instructions", params=params)).json()[
            "content"
        ] == "Clinic help"
        assert (
            await client.post("/artifacts/instructions/approve/1", params=params)
        ).status_code == 200
        assert (
            await client.patch(
                "/artifacts/instructions", params=params, json={"content": "Clinic v2"}
            )
        ).status_code == 200
        history = await client.get("/artifacts/instructions/history", params=params)
        assert [row["content"] for row in history.json()] == ["Clinic v2", "Clinic v1"]
        assert (
            await client.post("/artifacts/instructions/rollback/1", params=params)
        ).status_code == 200
        assert (await client.get("/artifacts/instructions", params=params)).json()[
            "content"
        ] == "Clinic v1"
        assert (
            await client.post(
                "/artifacts/instructions/actions",
                params=params,
                json={"action": "update", "content": "Clinic v3"},
            )
        ).status_code == 200
        assert (await client.get("/artifacts", params=params)).json()[0]["content"] == "Clinic v3"
        assert (await client.get("/artifacts")).json() == []
        assert (await artifacts.active_texts())["instructions"] == "Help"
        assert (
            await client.patch("/artifacts/policies", params=params, json={"content": "Changed"})
        ).status_code == 403
        assert (await client.delete("/artifacts/instructions", params=params)).status_code == 200
        assert (await client.get("/artifacts/instructions", params=params)).json()[
            "content"
        ] == "Clinic help"

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("get", "/artifacts", None),
            ("get", "/artifacts/profile", None),
            ("get", "/artifacts/instructions", None),
            ("get", "/artifacts/instructions/history", None),
            ("post", "/artifacts/instructions/propose", {"content": "x"}),
            ("patch", "/artifacts/instructions", {"content": "x"}),
            ("post", "/artifacts/instructions/approve/1", None),
            ("post", "/artifacts/instructions/rollback/1", None),
            ("post", "/artifacts/instructions/actions", {"action": "update", "content": "x"}),
            ("delete", "/artifacts/instructions", None),
        ],
    )
    async def test_unknown_profile_is_404_without_default_fallback(
        self, client, method, path, body
    ):
        response = await client.request(method, path, params={"profile": "missing"}, json=body)
        assert response.status_code == 404
        assert "Unknown assistant profile" in response.json()["detail"]


class TestProposals:
    async def test_proposals_across_profiles_with_diffs(self, artifacts):
        other = AssistantProfile(
            name="support",
            artifacts=(ArtifactDefinition(name="instructions", default="Assist"),),
        )
        service = ArtifactService(ArtifactsConfig(), PROFILE, profiles=[other])
        await service.start()
        await service.propose("instructions", "Help more", actor=Actor("assistant"))
        await service.for_profile("support").propose(
            "instructions", "Assist more", actor=Actor("assistant"), rationale="Warmer"
        )
        app = FastAPI()
        app.include_router(router)
        app.include_router(proposals_router)
        app.state.artifact_service = service
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            everything = (await c.get("/artifact-proposals")).json()
            support = (await c.get("/artifact-proposals", params={"profile": "support"})).json()
            none_rejected = (await c.get("/artifact-proposals?status=rejected")).json()
            first = (await c.get("/artifact-proposals", params={"limit": 1})).json()
            second = (
                await c.get(
                    "/artifact-proposals", params={"limit": 1, "before_id": first["next_before"]}
                )
            ).json()
        # Newest first across profiles: the support proposal was made last.
        assert [(r["profile"], r["version"]) for r in everything["proposals"]] == [
            ("support", 1),
            ("shop", 1),
        ]
        assert everything["next_before"] is None
        assert [r["rationale"] for r in support["proposals"]] == ["Warmer"]
        assert "+Assist more" in support["proposals"][0]["diff"]
        assert none_rejected == {"proposals": [], "next_before": None}
        # One record per page, and the cursor reaches the older one.
        assert [r["profile"] for r in first["proposals"]] == ["support"]
        assert [r["profile"] for r in second["proposals"]] == ["shop"]
        assert second["next_before"] is None

    async def test_version_record(self, client, artifacts):
        await artifacts.propose("instructions", "Help more", actor=Actor("assistant"))
        body = (await client.get("/artifacts/instructions/versions/1")).json()
        assert body["status"] == "pending"
        assert body["active_content"] == "Help"
        assert "+Help more" in body["diff"]
        assert (await client.get("/artifacts/instructions/versions/5")).status_code == 404

    async def test_propose_with_rationale_then_reject(self, client, artifacts):
        response = await client.post(
            "/artifacts/instructions/propose", json={"content": "New", "rationale": "Clearer"}
        )
        assert response.status_code == 201
        assert response.json()["rationale"] == "Clearer"
        response = await client.post("/artifacts/instructions/reject/1", json={"reason": "Not now"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "rejected"
        assert body["decision_reason"] == "Not now"
        assert body["effective_on_next_request"] is False
        assert (await artifacts.active_texts())["instructions"] == "Help"
        again = await client.post("/artifacts/instructions/reject/1")
        assert again.status_code == 409

    async def test_reject_through_the_actions_endpoint(self, client, artifacts):
        await artifacts.propose("instructions", "New", actor=Actor("assistant"))
        response = await client.post(
            "/artifacts/instructions/actions",
            json={"action": "reject", "version": 1, "reason": "No"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "rejected"

    async def test_an_artifact_named_proposals_stays_readable(self):
        named = AssistantProfile(
            name="shop", artifacts=(ArtifactDefinition(name="proposals", default="Offer list"),)
        )
        service = ArtifactService(ArtifactsConfig(), named)
        await service.start()
        app = FastAPI()
        app.include_router(router)
        app.include_router(proposals_router)
        app.state.artifact_service = service
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            body = (await c.get("/artifacts/proposals")).json()
        assert body["content"] == "Offer list"


class TestSubjects:
    @pytest.fixture
    async def tracker_client(self):
        from assistant_runtime.app.routes.artifacts import subjects_router

        profile = AssistantProfile(
            name="tracker",
            artifacts=(
                ArtifactDefinition(name="instructions", required=True, default="Track"),
                ArtifactDefinition(name="progress", scope="subject", default="Nothing yet"),
            ),
        )
        service = ArtifactService(ArtifactsConfig(), profile)
        await service.start()
        app = FastAPI()
        app.include_router(router)
        app.include_router(subjects_router)
        app.state.artifact_service = service
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c

    async def test_subject_artifacts_need_and_take_a_subject(self, tracker_client):
        c = tracker_client
        assert (await c.get("/artifacts/progress")).status_code == 422
        body = (await c.get("/artifacts/progress", params={"subject": "agent-a"})).json()
        assert (body["content"], body["source"]) == ("Nothing yet", "default")
        response = await c.patch(
            "/artifacts/progress", params={"subject": "agent-a"}, json={"content": "Busy"}
        )
        assert response.status_code == 200
        assert response.json()["subject"] == "agent-a"
        other = (await c.get("/artifacts/progress", params={"subject": "agent-b"})).json()
        assert other["content"] == "Nothing yet"
        assert (await c.get("/artifact-subjects")).json() == {
            "profile": "tracker",
            "subjects": ["agent-a"],
        }
        assert (
            await c.get("/artifacts/progress", params={"subject": "bad subject"})
        ).status_code == 422

    async def test_profile_describes_scope_and_includes(self, tracker_client):
        body = (await tracker_client.get("/artifacts/profile")).json()
        assert [a["scope"] for a in body["artifacts"]] == ["profile", "subject"]
        assert body["include"] == []
        assert body["collections"] == []


class TestPromptPreview:
    async def test_preview_returns_the_record_with_content(self):
        from types import SimpleNamespace

        from assistant_runtime.app.routes.artifacts import prompt_router

        calls = []

        async def preview_prompt(profile, subject):
            calls.append((profile, subject))
            return {"profile": profile, "subject": subject, "content": "Help"}

        app = FastAPI()
        app.include_router(prompt_router)
        app.state.assistant_service = SimpleNamespace(preview_prompt=preview_prompt)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            body = (
                await c.get("/artifact-prompt", params={"profile": "lead", "subject": "agent-a"})
            ).json()
        assert body["content"] == "Help"
        assert calls == [("lead", "agent-a")]


class TestHostLabels:
    async def test_a_label_records_who_in_the_host_wrote(self, client, artifacts):
        written = await client.patch(
            "/artifacts/instructions", json={"content": "Help more", "label": "owner"}
        )
        assert written.json()["proposed_by"] == "local:owner"
        await artifacts.propose("instructions", "Proposal", actor=Actor("assistant"))
        approved = await client.post("/artifacts/instructions/approve/2", json={"label": "watcher"})
        assert approved.json()["decided_by"] == "local:watcher"
        plain = await client.patch("/artifacts/instructions", json={"content": "Plain"})
        assert plain.json()["proposed_by"] == "local"

    async def test_a_label_must_be_a_short_name(self, client):
        response = await client.patch(
            "/artifacts/instructions", json={"content": "x", "label": "Not A Label"}
        )
        assert response.status_code == 422
