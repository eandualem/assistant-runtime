"""In-process domain events: something the runtime recorded, for the host to act on.

Services publish dicts with a ``type`` (for example ``artifact_proposal``)
after the change is stored; a host embedding the runtime subscribes through
``app.state.events``. Delivery is at most once and in process: a handler
that misses an event (a restart, an exception) recovers from the records
the event names, through the service or its routes.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

Handler = Callable[[dict[str, Any]], Awaitable[None] | None]


class EventHub:
    """Publish domain events to the handlers subscribed at the time."""

    def __init__(self) -> None:
        self._handlers: list[Handler] = []

    def subscribe(self, handler: Handler) -> Callable[[], None]:
        """Call ``handler`` (sync or async) with every later event; returns an unsubscribe."""
        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    async def publish(self, event: dict[str, Any]) -> None:
        """Deliver ``event`` to each handler in turn; a failing handler is logged, never raised.

        The publisher awaits the handlers, so a handler should hand slow work
        to its own task.
        """
        for handler in list(self._handlers):
            try:
                result = handler(copy.deepcopy(event))  # handlers never see each other's edits
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.warning("Event handler failed", event_type=event.get("type"), error=str(e))
