"""Protocol definitions for cross-module boundaries."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LifecycleAware(Protocol):
    """Any component that participates in application lifecycle."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def health_check(self) -> dict: ...
