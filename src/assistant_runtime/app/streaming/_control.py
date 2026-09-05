"""Ownership and native cancellation for one accepted application turn."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import CancellationToken
from pydantic_ai.exceptions import RunCancelled


@dataclass
class TurnControl:
    """Keep cancellation and finalization alive independently of the transport."""

    token: CancellationToken = field(default_factory=CancellationToken)
    events: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    setup_task: asyncio.Future[Any] | None = None
    error: BaseException | None = None
    accepting_cancel: bool = True

    def cancel(self) -> bool:
        """Request native cancellation without interrupting persistence or delivery."""
        if not self.accepting_cancel or self.done.is_set():
            return False
        self.token.cancel()
        if self.setup_task is not None:
            self.setup_task.cancel()
        return True

    def check_cancelled(self) -> None:
        """Stop before another phase begins if cancellation has been requested."""
        if self.token.cancelled:
            raise RunCancelled("Request cancelled")

    async def prepare(self, *operations: Awaitable[Any]) -> list[Any]:
        """Run setup concurrently, draining every child on failure or cancellation."""
        tasks = [asyncio.ensure_future(operation) for operation in operations]
        self.setup_task = asyncio.gather(*tasks)
        if self.token.cancelled:
            self.setup_task.cancel()
        try:
            return await self.setup_task
        except asyncio.CancelledError:
            if self.token.cancelled:
                raise RunCancelled("Request cancelled during setup") from None
            raise
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self.setup_task = None
