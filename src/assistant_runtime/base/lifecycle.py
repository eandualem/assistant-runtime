"""Lifecycle manager — ordered startup, shutdown, and health aggregation."""

import asyncio

from loguru import logger

from assistant_runtime.base.protocols import LifecycleAware


class LifecycleManager:
    """Manages ordered startup, shutdown, and health for all registered components."""

    def __init__(self) -> None:
        self._components: dict[str, LifecycleAware] = {}
        self._started: list[str] = []

    async def register(self, name: str, component: LifecycleAware) -> None:
        """Register a component. Registration order = startup order."""
        if name in self._components:
            raise ValueError(f"Component '{name}' already registered")
        self._components[name] = component
        logger.info("Component registered", component=name)

    async def start_all(self) -> None:
        """Start all components in registration order. Rolls back on failure."""
        for name, component in self._components.items():
            # Own resources as soon as startup begins, including partial acquisition.
            self._started.append(name)
            try:
                await component.start()
                logger.info("Component started", component=name)
            except (Exception, asyncio.CancelledError) as e:
                logger.error("Component failed to start", component=name, error=repr(e))
                await self.stop_all()
                raise

    async def stop_all(self) -> None:
        """Stop all started components in reverse order."""
        cleanup = asyncio.create_task(self._stop_all())
        cancelled = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                cancelled = exc
        cleanup.result()
        if cancelled is not None:
            raise cancelled

    async def _stop_all(self) -> None:
        for name in reversed(self._started):
            try:
                await self._components[name].stop()
                logger.info("Component stopped", component=name)
            except Exception as e:
                logger.error("Component failed to stop", component=name, error=str(e))
        self._started.clear()

    async def health(self) -> dict:
        """Aggregate health from all components.

        ``status`` is ``unhealthy`` when a component is not healthy,
        ``degraded`` when all are healthy but one reports ``ready: false``,
        and ``ok`` otherwise.
        """
        results = {}
        overall_healthy = True
        ready = True
        for name, component in self._components.items():
            try:
                check = await component.health_check()
                results[name] = check
                if not check.get("healthy", False):
                    overall_healthy = False
                if not check.get("ready", True):
                    ready = False
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                overall_healthy = False
        status = "unhealthy" if not overall_healthy else "ok" if ready else "degraded"
        return {"healthy": overall_healthy, "status": status, "components": results}
