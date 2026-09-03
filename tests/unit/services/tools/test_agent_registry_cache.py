"""Tests for the AgentRegistryCache class."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from assistant_runtime.services.tools._agent_registry_cache import (
    AgentRegistryCache,
    _reset_registry_cache,
    get_registry_cache,
)

MODULE = "assistant_runtime.services.tools._agent_registry_cache"

SAMPLE_AGENTS = [
    {
        "name": "leo",
        "display_name": "Leo",
        "role": "Strategy Co-Architect",
        "session": "leo",
        "type": "entity",
        "home": "/srv/agents/leo",
    },
    {
        "name": "ike",
        "display_name": "Ike",
        "role": "Core Orchestrator",
        "session": "ike",
        "type": "entity",
        "home": "/srv/agents/ike",
    },
    {
        "name": "agent-backbone",
        "display_name": "Agent Backbone",
        "role": "Implementation",
        "session": "agent-backbone",
        "type": "coding-agent",
        "home": "/srv/agents/agent-backbone",
    },
]


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton between tests."""
    _reset_registry_cache()
    yield
    _reset_registry_cache()


def _mock_backbone_success(items: list | None = None):
    """Return an AsyncMock that simulates a successful backbone response."""
    if items is None:
        items = SAMPLE_AGENTS
    return AsyncMock(return_value=(200, {"items": items, "total": len(items)}))


def _mock_backbone_error(status: int = -1, body: dict | None = None):
    """Return an AsyncMock that simulates a backbone error response."""
    if body is None:
        body = {"error": "Connection refused", "error_code": "BACKBONE_HTTP_ERROR"}
    return AsyncMock(return_value=(status, body))


# ---------------------------------------------------------------------------
# TestAgentRegistryCache
# ---------------------------------------------------------------------------


class TestAgentRegistryCache:
    @patch(f"{MODULE}.backbone_request")
    async def test_get_agents_fetches_on_first_call(self, mock_backbone):
        mock_backbone.return_value = (200, {"items": SAMPLE_AGENTS, "total": 3})

        cache = AgentRegistryCache()
        result = await cache.get_agents()

        assert result == SAMPLE_AGENTS
        mock_backbone.assert_awaited_once_with("GET", "/api/agents")

    @patch(f"{MODULE}.backbone_request")
    async def test_get_agents_returns_cached_on_second_call(self, mock_backbone):
        mock_backbone.return_value = (200, {"items": SAMPLE_AGENTS, "total": 3})

        cache = AgentRegistryCache()
        first = await cache.get_agents()
        second = await cache.get_agents()

        assert first == SAMPLE_AGENTS
        assert second == SAMPLE_AGENTS
        mock_backbone.assert_awaited_once()

    @patch(f"{MODULE}.backbone_request")
    async def test_refresh_clears_on_failure(self, mock_backbone):
        # First call succeeds to populate cache
        mock_backbone.return_value = (200, {"items": SAMPLE_AGENTS, "total": 3})
        cache = AgentRegistryCache()
        await cache.get_agents()
        assert cache._agents is not None

        # Second call (refresh) fails
        mock_backbone.return_value = (500, {"error": "Internal Server Error"})
        await cache.refresh()

        # Cache should be cleared — get_agents returns None (because refresh
        # already ran and set _agents to None, and the next get_agents will
        # try to refetch which also fails)
        assert await cache.get_agents() is None

    @patch(f"{MODULE}.backbone_request")
    async def test_ttl_expiry_triggers_refetch(self, mock_backbone):
        mock_backbone.return_value = (200, {"items": SAMPLE_AGENTS, "total": 3})

        cache = AgentRegistryCache(ttl=0.01)
        await cache.get_agents()
        assert mock_backbone.await_count == 1

        # Wait for TTL to expire
        await asyncio.sleep(0.02)

        await cache.get_agents()
        assert mock_backbone.await_count == 2

    @patch(f"{MODULE}.backbone_request")
    async def test_backbone_unavailable_returns_none(self, mock_backbone):
        mock_backbone.return_value = (
            -1,
            {"error": "Connection refused", "error_code": "BACKBONE_HTTP_ERROR"},
        )

        cache = AgentRegistryCache()
        result = await cache.get_agents()

        assert result is None

    @patch(f"{MODULE}.backbone_request")
    async def test_missing_items_key_returns_none(self, mock_backbone):
        mock_backbone.return_value = (200, {"total": 5})

        cache = AgentRegistryCache()
        result = await cache.get_agents()

        assert result is None


# ---------------------------------------------------------------------------
# TestGetAvailableSessions
# ---------------------------------------------------------------------------


class TestGetAvailableSessions:
    async def test_returns_session_names(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        sessions = cache.get_available_sessions()

        assert sessions == ["leo", "ike", "agent-backbone"]

    async def test_empty_when_no_cache(self):
        cache = AgentRegistryCache()

        sessions = cache.get_available_sessions()

        assert sessions == []


# ---------------------------------------------------------------------------
# TestGetAgentInfo
# ---------------------------------------------------------------------------


class TestGetAgentInfo:
    async def test_finds_agent_by_session(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        info = cache.get_agent_info("ike")

        assert info is not None
        assert info["name"] == "ike"
        assert info["role"] == "Core Orchestrator"

    async def test_returns_none_for_unknown(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        info = cache.get_agent_info("nonexistent-session")

        assert info is None

    async def test_returns_none_when_cache_empty(self):
        cache = AgentRegistryCache()

        info = cache.get_agent_info("leo")

        assert info is None


# ---------------------------------------------------------------------------
# TestGetWorkingDirectory
# ---------------------------------------------------------------------------


class TestGetWorkingDirectory:
    async def test_returns_home_field(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        home = cache.get_working_directory("leo")

        assert home == "/srv/agents/leo"

    async def test_returns_none_when_no_home(self):
        cache = AgentRegistryCache()
        cache._agents = [{"name": "minimal", "session": "minimal", "type": "entity"}]

        home = cache.get_working_directory("minimal")

        assert home is None

    async def test_returns_none_for_unknown_session(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        home = cache.get_working_directory("nonexistent")

        assert home is None


# ---------------------------------------------------------------------------
# TestBuildEcosystemLines
# ---------------------------------------------------------------------------


class TestBuildEcosystemLines:
    async def test_builds_lines_from_agents(self):
        cache = AgentRegistryCache()
        cache._agents = SAMPLE_AGENTS

        lines = cache.build_ecosystem_lines()

        assert len(lines) == 4  # header + 3 agents
        assert lines[0] == "Agents and systems you work with:"
        assert "Leo" in lines[1]
        assert "Strategy Co-Architect" in lines[1]
        assert "entity" in lines[1]
        assert "[session: leo]" in lines[1]
        assert "Ike" in lines[2]
        assert "Agent Backbone" in lines[3]
        assert "coding-agent" in lines[3]

    async def test_empty_when_no_cache(self):
        cache = AgentRegistryCache()

        lines = cache.build_ecosystem_lines()

        assert lines == []


# ---------------------------------------------------------------------------
# TestModuleSingleton
# ---------------------------------------------------------------------------


class TestModuleSingleton:
    async def test_get_registry_cache_returns_same_instance(self):
        first = get_registry_cache()
        second = get_registry_cache()

        assert first is second

    async def test_reset_clears_instance(self):
        first = get_registry_cache()
        _reset_registry_cache()
        second = get_registry_cache()

        assert first is not second
