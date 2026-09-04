"""Tests for application lifespan wiring."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.main import create_app, lifespan


@pytest.mark.asyncio
async def test_lifespan_injects_runtime_settings_into_services(monkeypatch):
    async def _register_database(app_state, lifecycle):
        app_state.database_service = SimpleNamespace(_healthy=True)

    async def _register_llm(app_state, lifecycle):
        return None

    async def _register_history(app_state, lifecycle):
        return None

    async def _register_media(app_state, lifecycle):
        return None

    async def _register_oauth(app_state, lifecycle):
        return None

    async def _register_tools(app_state, lifecycle):
        return None

    async def _register_assistant(app_state, lifecycle):
        app_state.assistant_service = SimpleNamespace(
            _runtime_settings=None,
            cleanup_expired_sessions=AsyncMock(return_value=0),
            set_runtime_settings=MagicMock(),
        )

    async def _register_heartbeat(app_state, lifecycle):
        app_state.heartbeat_service = SimpleNamespace()

    async def _register_streaming(app_state, lifecycle):
        app_state.streaming_service = SimpleNamespace(attach_ingress=MagicMock())

    async def _register_ingress(app_state, lifecycle):
        app_state.ingress_service = SimpleNamespace()

    monkeypatch.setattr("assistant_runtime.main.register_database", _register_database)
    monkeypatch.setattr("assistant_runtime.main.register_llm", _register_llm)
    monkeypatch.setattr("assistant_runtime.main.register_history", _register_history)
    monkeypatch.setattr("assistant_runtime.main.register_media", _register_media)
    monkeypatch.setattr("assistant_runtime.main.register_oauth", _register_oauth)
    monkeypatch.setattr("assistant_runtime.main.register_tools", _register_tools)
    monkeypatch.setattr("assistant_runtime.main.register_assistant", _register_assistant)
    monkeypatch.setattr("assistant_runtime.main.register_heartbeat", _register_heartbeat)
    monkeypatch.setattr("assistant_runtime.main.register_streaming", _register_streaming)
    monkeypatch.setattr("assistant_runtime.main.register_ingress", _register_ingress)
    monkeypatch.setattr("assistant_runtime.main.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("assistant_runtime.main.setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "assistant_runtime.main.RuntimeSettings.load_from_db",
        AsyncMock(return_value=None),
    )

    app = create_app()
    async with lifespan(app):
        app.state.assistant_service.set_runtime_settings.assert_called_once_with(
            app.state.runtime_settings
        )
