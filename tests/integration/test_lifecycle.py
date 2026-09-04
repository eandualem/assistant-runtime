"""Integration tests for lifecycle management — startup, shutdown, health."""

from __future__ import annotations

import pytest

from assistant_runtime.app.assistant.interface import AssistantService
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.services.history.interface import HistoryService
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.interface import ToolService


class TestFullLifecycle:
    @pytest.mark.asyncio
    async def test_start_all_services(self, wired_services):
        """All services start successfully in dependency order."""
        health = await wired_services["lifecycle"].health()
        assert health["healthy"] is True
        assert len(health["components"]) == 5

    @pytest.mark.asyncio
    async def test_all_components_healthy(self, wired_services):
        """Each individual component reports healthy."""
        health = await wired_services["lifecycle"].health()
        for name, component_health in health["components"].items():
            assert component_health["healthy"] is True, f"{name} not healthy"

    @pytest.mark.asyncio
    async def test_stop_all_services(self, wired_services):
        """Stopping all services sets them to not-started."""
        await wired_services["lifecycle"].stop_all()
        # After stop, health checks should report unhealthy
        assistant = wired_services["assistant_service"]
        health = await assistant.health_check()
        assert health["healthy"] is False

    @pytest.mark.asyncio
    async def test_lifecycle_order(self, monkeypatch):
        """Services start in registration order (LLM → history → tools → assistant)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-order")
        settings = AppSettings()
        lm = LifecycleManager()

        start_order = []

        class TrackingLlm(LlmService):
            async def start(self):
                start_order.append("llm")
                await super().start()

        class TrackingHistory(HistoryService):
            async def start(self):
                start_order.append("history")
                await super().start()

        class TrackingTools(ToolService):
            async def start(self):
                start_order.append("tools")
                await super().start()

        llm = TrackingLlm(config=settings.llm)
        history = TrackingHistory(config=settings.history, llm_service=llm)
        tools = TrackingTools(config=settings.tools)
        assistant = AssistantService(
            config=settings.assistant,
            llm_service=llm,
            history_service=history,
            tool_service=tools,
        )

        await lm.register("llm", llm)
        await lm.register("history", history)
        await lm.register("tools", tools)
        await lm.register("assistant", assistant)
        await lm.start_all()

        assert start_order == ["llm", "history", "tools"]  # assistant start is plain
        await lm.stop_all()


class TestConfigComposition:
    def test_app_settings_has_all_configs(self):
        """AppSettings composes all module configs."""
        settings = AppSettings()
        assert hasattr(settings, "llm")
        assert hasattr(settings, "history")
        assert hasattr(settings, "tools")
        assert hasattr(settings, "assistant")
        assert hasattr(settings, "streaming")

    def test_config_defaults_valid(self):
        """All module configs have valid defaults (no ValidationError)."""
        settings = AppSettings()
        assert settings.llm.primary_model is not None
        assert settings.history.token_budget > 0
        assert settings.tools.max_tools_per_request > 0
        assert settings.assistant.max_turns > 0
        assert settings.streaming.stream_timeout_seconds > 0


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, monkeypatch):
        """If a service fails to start, previously started services are stopped."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-rollback")
        settings = AppSettings()
        lm = LifecycleManager()

        stop_called = []

        class TrackingLlm(LlmService):
            async def stop(self):
                stop_called.append("llm")
                await super().stop()

        class FailingTools(ToolService):
            async def start(self):
                raise RuntimeError("Tool service exploded")

        llm = TrackingLlm(config=settings.llm)
        tools = FailingTools(config=settings.tools)

        await lm.register("llm", llm)
        await lm.register("tools", tools)

        with pytest.raises(RuntimeError, match="exploded"):
            await lm.start_all()

        # LLM was started before tools failed, so it should have been stopped
        assert "llm" in stop_called
