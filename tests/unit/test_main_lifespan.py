"""Tests for application lifespan wiring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.main import create_app, create_runtime, lifespan


@pytest.mark.asyncio
async def test_lifespan_injects_runtime_settings_into_services(monkeypatch):
    async def _register_database(app_state, lifecycle, *, settings=None):
        app_state.database_service = SimpleNamespace(_healthy=True)

    async def _register_llm(app_state, lifecycle, *, settings=None):
        return None

    async def _register_history(app_state, lifecycle, *, settings=None):
        return None

    async def _register_media(app_state, lifecycle, *, settings=None):
        return None

    async def _register_oauth(app_state, lifecycle, *, settings=None):
        return None

    async def _register_tools(app_state, lifecycle, *, settings=None):
        return None

    async def _register_assistant(app_state, lifecycle, *, settings=None):
        app_state.assistant_service = SimpleNamespace(
            _runtime_settings=None,
            cleanup_expired_sessions=AsyncMock(return_value=0),
            set_runtime_settings=MagicMock(),
        )

    async def _register_heartbeat(app_state, lifecycle, *, settings=None):
        app_state.heartbeat_service = SimpleNamespace()

    async def _register_streaming(app_state, lifecycle, *, settings=None):
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


@pytest.mark.parametrize("stage", ["component-start", "runtime-settings"])
@pytest.mark.parametrize("cancel", [False, True], ids=["error", "cancelled"])
async def test_runtime_cleans_up_when_startup_does_not_finish(monkeypatch, stage, cancel):
    reached_failure = asyncio.Event()
    release_failure = asyncio.Event()
    started = []
    stopped = []

    async def fail():
        reached_failure.set()
        await release_failure.wait()
        raise RuntimeError("Startup failed")

    class Component:
        def __init__(self, name):
            self.name = name

        async def start(self):
            if self.name == "third" and stage == "component-start":
                await fail()
            started.append(self.name)

        async def stop(self):
            stopped.append(self.name)

    async def register_components(app_state, lifecycle, **kwargs):
        for name in ("first", "second", "third"):
            await lifecycle.register(name, Component(name))

    for name in (
        "database",
        "oauth",
        "llm",
        "history",
        "media",
        "mcp",
        "tools",
        "assistant",
        "streaming",
        "ingress",
        "heartbeat",
    ):
        monkeypatch.setattr(f"assistant_runtime.main.register_{name}", AsyncMock())
    monkeypatch.setattr("assistant_runtime.main.register_database", register_components)
    monkeypatch.setattr("assistant_runtime.main.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr("assistant_runtime.main.setup_logging", lambda **kwargs: None)
    monkeypatch.setattr("assistant_runtime.main.initialize_tracing", lambda: True)
    shutdown = MagicMock()
    monkeypatch.setattr("assistant_runtime.main.shutdown_tracing", shutdown)
    if stage == "runtime-settings":
        monkeypatch.setattr(
            "assistant_runtime.main.RuntimeSettings.load_from_db", AsyncMock(side_effect=fail)
        )

    async def open_runtime():
        async with create_runtime():
            pytest.fail("Startup should not complete")

    task = asyncio.create_task(open_runtime())
    try:
        async with asyncio.timeout(5):
            await reached_failure.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release_failure.set()
                with pytest.raises(RuntimeError, match="Startup failed"):
                    await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert started == (
        ["first", "second"] if stage == "component-start" else ["first", "second", "third"]
    )
    assert stopped == list(reversed(started))
    shutdown.assert_called_once_with()
