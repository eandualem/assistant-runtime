"""/artifacts routes against a real ArtifactService on an in-memory store."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.artifacts import router
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
