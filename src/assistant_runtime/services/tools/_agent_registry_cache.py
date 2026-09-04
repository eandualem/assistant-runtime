"""Cached client for the backbone's agent registry API.

Fetches agent data from GET /api/agents and caches with a 5-minute TTL.
Module-level singleton accessed via get_registry_cache().
"""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from assistant_runtime.services.tools._backbone_client import backbone_request

_DEFAULT_TTL = 300.0  # 5 minutes


class AgentRegistryCache:
    """Cached wrapper around the backbone's GET /api/agents endpoint."""

    def __init__(self, ttl: float = _DEFAULT_TTL) -> None:
        self._agents: list[dict[str, Any]] | None = None
        self._fetched_at: float = 0.0
        self._ttl = ttl

    def _is_stale(self) -> bool:
        return self._agents is None or (time.monotonic() - self._fetched_at) >= self._ttl

    async def get_agents(self) -> list[dict[str, Any]] | None:
        """Return cached agents list. Fetches from backbone if stale/empty.

        Returns None if the backbone is unavailable.
        """
        if self._is_stale():
            await self.refresh()
        return self._agents

    async def refresh(self) -> bool:
        """Force refresh from backbone. Returns True on success."""
        status, body = await backbone_request("GET", "/api/agents")
        if status != 200 or not isinstance(body, dict):
            logger.warning(
                "Failed to fetch agent registry from backbone",
                status=status,
                body=str(body)[:200],
            )
            self._agents = None
            return False

        items = body.get("items")
        if not isinstance(items, list):
            logger.warning(
                "Backbone /api/agents response missing 'items' list", body=str(body)[:200]
            )
            self._agents = None
            return False

        self._agents = items
        self._fetched_at = time.monotonic()
        logger.debug("Agent registry cache refreshed", agent_count=len(items))
        return True

    def get_agent_info(self, session: str) -> dict[str, Any] | None:
        """Look up a single agent by session name."""
        if self._agents is None:
            return None
        for agent in self._agents:
            if agent.get("session") == session:
                return agent
        return None

    def get_working_directory(self, session: str) -> str | None:
        """Resolve working directory (home field) for a session."""
        info = self.get_agent_info(session)
        if info is None:
            return None
        return info.get("home")

    def build_ecosystem_lines(self) -> list[str]:
        """Build ecosystem description lines for the prompt builder."""
        if self._agents is None:
            return []
        lines: list[str] = ["Agents and systems you work with:"]
        for agent in self._agents:
            display_name = agent.get("display_name") or agent.get("name", "Unknown")
            role = agent.get("role", "")
            agent_type = agent.get("type", "")
            session = agent.get("session", "")
            parts = [f"- {display_name}"]
            if role:
                parts[0] += f" ({role})"
            parts[0] += f": {agent_type}" if agent_type else ""
            if session:
                parts[0] += f" [session: {session}]"
            lines.append(parts[0])
        return lines


# --- Module-level singleton ---

_instance: AgentRegistryCache | None = None


def get_registry_cache() -> AgentRegistryCache:
    """Return the module-level singleton cache instance."""
    global _instance
    if _instance is None:
        _instance = AgentRegistryCache()
    return _instance


def _reset_registry_cache() -> None:
    """Reset the singleton (for testing only)."""
    global _instance
    _instance = None
