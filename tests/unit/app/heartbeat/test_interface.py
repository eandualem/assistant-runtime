"""Tests for HeartbeatService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from assistant_runtime.app.heartbeat.config import HeartbeatConfig
from assistant_runtime.app.heartbeat.exceptions import HeartbeatError
from assistant_runtime.app.heartbeat.interface import HeartbeatService


@pytest.fixture
def ingress():
    service = MagicMock()
    service.deliver = AsyncMock(return_value={"status": "delivered", "session_id": "s1"})
    return service


@pytest.fixture
def service(ingress):
    return HeartbeatService(
        config=HeartbeatConfig(enabled=False, interval_seconds=60, startup_delay_seconds=0),
        ingress_service=ingress,
    )


class TestLifecycle:
    async def test_disabled_by_default(self):
        assert HeartbeatConfig().enabled is False

    async def test_start_disabled_does_not_spawn_task(self, service):
        await service.start()
        assert service._task is None
        assert (await service.health_check())["healthy"] is True

    async def test_start_enabled_spawns_task_and_stop_cancels_it(self, ingress):
        service = HeartbeatService(
            config=HeartbeatConfig(enabled=True, interval_seconds=60, startup_delay_seconds=0),
            ingress_service=ingress,
        )
        await service.start()
        await asyncio.sleep(0.01)
        assert service._task is not None
        assert not service._task.done()
        assert (await service.health_check())["task_running"] is True
        await service.stop()
        assert service._task is None


class TestEnqueueOnce:
    async def test_raises_when_not_started(self, service):
        with pytest.raises(HeartbeatError):
            await service.enqueue_once()

    async def test_delivers_the_check_in_through_the_ingress(self, service, ingress):
        await service.start()
        result = await service.enqueue_once()
        assert result["status"] == "delivered"
        assert ingress.deliver.await_args.kwargs == {
            "from_agent": "heartbeat",
            "via": "heartbeat",
            "message": "[via:heartbeat]",
        }
        assert (await service.health_check())["last_heartbeat_at"] is not None

    async def test_queued_result_is_not_a_heartbeat(self, service, ingress):
        ingress.deliver = AsyncMock(return_value={"status": "queued", "inbox_id": "x"})
        await service.start()
        await service.enqueue_once()
        assert (await service.health_check())["last_heartbeat_at"] is None


class TestRunLoop:
    async def test_run_loop_records_errors(self, ingress):
        ingress.deliver = AsyncMock(side_effect=RuntimeError("boom"))
        service = HeartbeatService(
            config=HeartbeatConfig(enabled=True, interval_seconds=60, startup_delay_seconds=0),
            ingress_service=ingress,
        )
        await service.start()
        await asyncio.sleep(0.02)
        assert (await service.health_check())["last_error"] == "boom"
        await service.stop()
