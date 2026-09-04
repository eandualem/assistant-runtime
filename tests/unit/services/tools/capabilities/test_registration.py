"""Tests for capability registration: a capability appears only with a provider."""

from __future__ import annotations

from unittest.mock import MagicMock

from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.capabilities import CAPABILITIES, register_capabilities
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.interface import ToolService
from assistant_runtime.services.tools.providers.filesystem import MarkdownNotes


class TestRegisterCapabilities:
    def test_no_providers_registers_nothing(self):
        registry = ToolRegistry(ToolConfig())
        assert register_capabilities(registry, {}) == []
        assert registry.backend_tool_count() == 0

    def test_each_provider_registers_its_capability(self, tmp_path):
        registry = ToolRegistry(ToolConfig())
        registered = register_capabilities(registry, {"notes": MarkdownNotes(tmp_path)})
        assert registered == ["notes"]
        assert registry.get_tool_names() == ["manage_notes"]

    def test_unknown_provider_keys_are_ignored(self):
        registry = ToolRegistry(ToolConfig())
        assert register_capabilities(registry, {"teleport": MagicMock()}) == []

    def test_every_capability_has_a_registrar(self):
        assert set(CAPABILITIES) == {
            "notes",
            "library",
            "peers",
            "rooms",
            "reminders",
            "activity",
            "workgroups",
            "repositories",
            "approvals",
        }


class TestToolServiceProviders:
    async def test_unconfigured_capabilities_are_not_offered(self):
        service = ToolService(config=ToolConfig())
        await service.start()
        names = set(service.get_available_tools().tool_names)
        assert "get_time" in names
        assert "manage_notes" not in names
        assert "list_documents" not in names

    async def test_configured_capability_is_offered(self, tmp_path):
        service = ToolService(config=ToolConfig(), providers={"notes": MarkdownNotes(tmp_path)})
        await service.start()
        assert "manage_notes" in service.get_available_tools().tool_names
