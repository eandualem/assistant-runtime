"""Tests for the LifecycleManager."""

import pytest

from lovely_assistant.base.lifecycle import LifecycleManager


class FakeComponent:
    """A fake lifecycle-aware component for testing."""

    def __init__(self, name: str, *, fail_start: bool = False, fail_stop: bool = False):
        self.name = name
        self.started = False
        self.stopped = False
        self._fail_start = fail_start
        self._fail_stop = fail_stop

    async def start(self) -> None:
        if self._fail_start:
            raise RuntimeError(f"{self.name} failed to start")
        self.started = True

    async def stop(self) -> None:
        if self._fail_stop:
            raise RuntimeError(f"{self.name} failed to stop")
        self.stopped = True

    async def health_check(self) -> dict:
        return {"healthy": self.started and not self.stopped}


async def test_register_and_start_all():
    lm = LifecycleManager()
    c1 = FakeComponent("first")
    c2 = FakeComponent("second")

    await lm.register("first", c1)
    await lm.register("second", c2)
    await lm.start_all()

    assert c1.started
    assert c2.started


async def test_stop_all_reverses_order():
    lm = LifecycleManager()
    stop_order = []

    class TrackingComponent(FakeComponent):
        async def stop(self) -> None:
            stop_order.append(self.name)

    c1 = TrackingComponent("first")
    c2 = TrackingComponent("second")
    c3 = TrackingComponent("third")

    await lm.register("first", c1)
    await lm.register("second", c2)
    await lm.register("third", c3)
    await lm.start_all()
    await lm.stop_all()

    assert stop_order == ["third", "second", "first"]


async def test_health_aggregation():
    lm = LifecycleManager()
    c1 = FakeComponent("healthy_one")
    c2 = FakeComponent("healthy_two")

    await lm.register("one", c1)
    await lm.register("two", c2)
    await lm.start_all()

    result = await lm.health()
    assert result["healthy"] is True
    assert len(result["components"]) == 2


async def test_health_reports_unhealthy_component():
    lm = LifecycleManager()

    class UnhealthyComponent:
        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def health_check(self) -> dict:
            return {"healthy": False, "error": "degraded"}

    await lm.register("bad", UnhealthyComponent())
    await lm.start_all()

    result = await lm.health()
    assert result["healthy"] is False
    assert result["components"]["bad"]["healthy"] is False


async def test_rollback_on_start_failure():
    lm = LifecycleManager()
    c1 = FakeComponent("good")
    c2 = FakeComponent("bad", fail_start=True)

    await lm.register("good", c1)
    await lm.register("bad", c2)

    with pytest.raises(RuntimeError, match="bad failed to start"):
        await lm.start_all()

    # c1 was started then stopped during rollback
    assert c1.started
    assert c1.stopped


async def test_duplicate_registration_raises():
    lm = LifecycleManager()
    c1 = FakeComponent("dup")

    await lm.register("thing", c1)
    with pytest.raises(ValueError, match="already registered"):
        await lm.register("thing", c1)


async def test_health_with_no_components():
    lm = LifecycleManager()
    result = await lm.health()
    assert result == {"healthy": True, "components": {}}
