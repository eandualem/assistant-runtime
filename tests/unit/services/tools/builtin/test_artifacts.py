"""The artifact management tool against a real ArtifactService on an in-memory store."""

from __future__ import annotations

import pytest

from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.interface import ArtifactService
from assistant_runtime.services.artifacts.models import Actor
from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.builtin.artifacts import (
    build_manage_artifacts,
    register_artifact_tools,
)
from assistant_runtime.services.tools.config import ToolConfig

PROFILE = AssistantProfile(
    name="shop",
    artifacts=(
        ArtifactDefinition(
            name="instructions", role="purpose", required=True, default="Help shoppers"
        ),
        ArtifactDefinition(
            name="persona",
            role="voice",
            default="Warm",
            policy=ArtifactPolicy(assistant_edit="autonomous", assistant_activate=True),
        ),
        ArtifactDefinition(name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")),
        ArtifactDefinition(name="policies", default="Refunds", policy=ArtifactPolicy("none")),
    ),
)


@pytest.fixture
async def artifacts() -> ArtifactService:
    service = ArtifactService(ArtifactsConfig(), PROFILE)
    await service.start()
    return service


@pytest.fixture
def manage(artifacts):
    return build_manage_artifacts(artifacts)


class TestDispatch:
    async def test_unknown_action(self, manage):
        result = await manage(action="approve")
        assert result["success"] is False
        assert result["error_code"] == "unknown_action"
        assert "activate" in result["error"]

    async def test_no_service(self):
        result = await build_manage_artifacts(None)(action="list")
        assert result == {
            "error": "Artifact service not available",
            "error_code": "artifacts_unavailable",
            "success": False,
        }

    async def test_name_required(self, manage):
        result = await manage(action="view")
        assert result["error_code"] == "missing_name"

    async def test_unknown_artifact(self, manage):
        result = await manage(action="view", name="soul")
        assert result["success"] is False
        assert result["error_code"] == "unknown_artifact"
        assert "instructions, persona, scratchpad, policies" in result["error"]


class TestReads:
    async def test_list_describes_the_profile_and_permissions(self, manage, artifacts):
        await artifacts.update("persona", "Direct", actor=Actor("host"))
        result = await manage(action="list")
        assert result["success"]
        assert result["count"] == 4
        assert result["durable"] is False
        by_name = {item["name"]: item for item in result["artifacts"]}
        assert by_name["instructions"]["allowed_actions"] == ["propose"]
        assert by_name["instructions"]["version"] is None
        assert by_name["instructions"]["required"] is True
        assert by_name["persona"]["version"] == 1
        assert by_name["persona"]["allowed_actions"] == ["propose", "update", "activate"]
        assert by_name["policies"]["allowed_actions"] == []
        assert [item["name"] for item in result["artifacts"]] == list(PROFILE.names)

    async def test_view_default_then_store(self, manage):
        result = await manage(action="view", name="persona")
        assert result == {
            "name": "persona",
            "version": None,
            "content": "Warm",
            "source": "default",
            "success": True,
        }
        await manage(action="update", name="persona", content="Direct")
        result = await manage(action="view", name="persona")
        assert result["content"] == "Direct"
        assert result["source"] == "store"
        assert result["version"] == 1
        assert result["is_active"] is True

    async def test_history(self, manage):
        await manage(action="update", name="scratchpad", content="a")
        await manage(action="update", name="scratchpad", content="b")
        result = await manage(action="history", name="scratchpad")
        assert [(v["version"], v["is_active"]) for v in result["versions"]] == [
            (2, True),
            (1, False),
        ]
        assert result["count"] == 2


class TestWrites:
    async def test_propose_stays_inactive(self, manage, artifacts):
        result = await manage(action="propose", name="instructions", content="New")
        assert result["success"]
        assert result["activated"] is False
        assert result["version"] == 1
        assert result["live_version"] is None
        assert "inactive" in result["message"]
        assert (await artifacts.active_texts())["instructions"] == "Help shoppers"

    async def test_review_required_artifact_cannot_be_updated_or_activated(self, manage):
        await manage(action="propose", name="instructions", content="New")
        result = await manage(action="update", name="instructions", content="Now")
        assert result["error_code"] == "artifact_permission_denied"
        result = await manage(action="activate", name="instructions", version=1)
        assert result["error_code"] == "artifact_permission_denied"

    async def test_autonomous_update_is_live_at_once(self, manage, artifacts):
        result = await manage(action="update", name="scratchpad", content="Remember")
        assert result["activated"] is True
        assert result["live_version"] == 1
        assert (await artifacts.active_texts())["scratchpad"] == "Remember"

    async def test_activate_where_allowed(self, manage):
        await manage(action="update", name="persona", content="v1")
        await manage(action="update", name="persona", content="v2")
        result = await manage(action="activate", name="persona", version=1)
        assert result["success"]
        assert result["live_version"] == 1
        result = await manage(action="activate", name="persona")
        assert result["error_code"] == "missing_version"
        result = await manage(action="activate", name="persona", version=9)
        assert result["error_code"] == "artifact_version_not_found"

    async def test_expected_version_conflict(self, manage):
        await manage(action="update", name="scratchpad", content="v1")
        result = await manage(action="update", name="scratchpad", content="v2", expected_version=0)
        assert result["error_code"] == "artifact_version_conflict"
        result = await manage(action="update", name="scratchpad", content="v2", expected_version=1)
        assert result["success"]
        assert result["live_version"] == 2

    async def test_duplicate_and_empty_content(self, manage):
        await manage(action="update", name="scratchpad", content="same")
        result = await manage(action="update", name="scratchpad", content="same")
        assert result["unchanged"] is True
        assert "nothing changed" in result["message"]
        result = await manage(action="update", name="scratchpad", content="")
        assert result["success"] is False
        assert result["error_code"] == "artifact_error"

    async def test_read_only_artifact(self, manage):
        result = await manage(action="propose", name="policies", content="x")
        assert result["error_code"] == "artifact_permission_denied"


class TestRegistration:
    async def test_description_comes_from_the_profile(self, artifacts):
        registry = ToolRegistry(ToolConfig())
        register_artifact_tools(registry, artifacts)
        tools = registry.get_available_tools().backend_tools
        definition = next(t for t in tools if t.name == "manage_artifacts")
        assert "persona (voice; you may propose, update, activate)" in definition.description
        assert "policies (no role given; you may only read)" in definition.description
        assert definition.parameters_schema["properties"]["action"]["enum"] == [
            "list",
            "view",
            "history",
            "propose",
            "update",
            "activate",
        ]
        assert "expected_version" in definition.parameters_schema["properties"]

    def test_registers_without_a_service(self):
        registry = ToolRegistry(ToolConfig())
        register_artifact_tools(registry, None)
        assert "manage_artifacts" in registry.get_tool_names()
