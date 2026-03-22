"""Tests for HeartbeatService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lovely_assistant.app.heartbeat.config import HeartbeatConfig
from lovely_assistant.app.heartbeat.exceptions import HeartbeatError
from lovely_assistant.app.heartbeat.interface import HeartbeatService


@pytest.fixture
def assistant_service():
    service = MagicMock()
    session_store = MagicMock()
    session_store.list_sessions = AsyncMock(return_value=[])
    service.get_session_store.return_value = session_store
    return service


@pytest.fixture
def database_service():
    return MagicMock()


@pytest.fixture
def service(assistant_service, database_service):
    return HeartbeatService(
        config=HeartbeatConfig(enabled=False, interval_seconds=60, startup_delay_seconds=0),
        assistant_service=assistant_service,
        database_service=database_service,
    )


class TestLifecycle:
    async def test_start_disabled_does_not_spawn_task(self, service):
        await service.start()
        assert service._started is True
        assert service._task is None

    async def test_start_enabled_spawns_task(self, assistant_service, database_service):
        svc = HeartbeatService(
            config=HeartbeatConfig(enabled=True, interval_seconds=60, startup_delay_seconds=60),
            assistant_service=assistant_service,
            database_service=database_service,
        )

        await svc.start()

        assert svc._task is not None
        await svc.stop()

    async def test_stop_cancels_running_task(self, assistant_service, database_service):
        svc = HeartbeatService(
            config=HeartbeatConfig(enabled=True, interval_seconds=60, startup_delay_seconds=60),
            assistant_service=assistant_service,
            database_service=database_service,
        )
        await svc.start()

        task = svc._task
        assert task is not None

        await svc.stop()

        assert task.cancelled() or task.done()


class TestEnqueueOnce:
    async def test_raises_when_not_started(self, service):
        with pytest.raises(HeartbeatError, match="not started"):
            await service.enqueue_once()

    async def test_skips_when_no_database(self, assistant_service):
        svc = HeartbeatService(
            config=HeartbeatConfig(enabled=False),
            assistant_service=assistant_service,
            database_service=None,
        )
        await svc.start()

        result = await svc.enqueue_once()

        assert result == {"status": "skipped", "reason": "database_unavailable"}

    async def test_skips_when_no_active_session(self, service):
        await service.start()

        result = await service.enqueue_once()

        assert result == {"status": "skipped", "reason": "no_active_session"}

    async def test_injects_heartbeat_into_latest_session(
        self, monkeypatch, assistant_service, database_service
    ):
        assistant_service.get_session_store.return_value.list_sessions = AsyncMock(
            return_value=[{"session_id": "sess-1"}]
        )
        svc = HeartbeatService(
            config=HeartbeatConfig(
                enabled=False,
                interval_seconds=60,
                startup_delay_seconds=0,
                message="[via:heartbeat]",
                from_agent="heartbeat",
            ),
            assistant_service=assistant_service,
            database_service=database_service,
        )
        await svc.start()

        inject = AsyncMock(return_value={"status": "delivered", "session_id": "sess-1"})
        monkeypatch.setattr(
            "lovely_assistant.app.heartbeat.interface.inject_inbox_message",
            inject,
        )

        result = await svc.enqueue_once()

        assert result["status"] == "delivered"
        inject.assert_awaited_once_with(
            db=database_service,
            assistant_service=assistant_service,
            from_agent="heartbeat",
            via="heartbeat",
            message="[via:heartbeat]",
            session_id="sess-1",
        )
        assert svc._last_heartbeat_at is not None


class TestHealthCheck:
    async def test_health_check_reports_disabled_service(self, service):
        await service.start()

        result = await service.health_check()

        assert result["healthy"] is True
        assert result["enabled"] is False
        assert result["task_running"] is False


class TestRunLoop:
    async def test_run_loop_records_errors(self, assistant_service, database_service):
        svc = HeartbeatService(
            config=HeartbeatConfig(enabled=True, interval_seconds=60, startup_delay_seconds=0),
            assistant_service=assistant_service,
            database_service=database_service,
        )

        calls = 0

        async def _enqueue():
            nonlocal calls
            calls += 1
            raise RuntimeError("boom")

        svc.enqueue_once = AsyncMock(side_effect=_enqueue)
        await svc.start()

        await asyncio.sleep(0)
        await svc.stop()

        assert calls >= 1
        assert svc._last_error == "boom"
