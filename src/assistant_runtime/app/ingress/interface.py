"""IngressService — messages from other systems, delivered into a session.

Another agent, a bot, a scheduler or the runtime's own heartbeat can hand
the assistant a message. It is delivered the way a person's steering would
be: queued into the session's running turn if one is live, otherwise run
as a turn of its own (in the background, with the events sent to the
session's Socket.IO room). When no session can be found the message waits
in the inbox and is drained into the next turn that starts. The model
answers where the envelope says the message came from.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.socketio_server import socket_event_name

if TYPE_CHECKING:
    from assistant_runtime.app.assistant.interface import AssistantService
    from assistant_runtime.app.streaming.interface import StreamingService
    from assistant_runtime.services.database.interface import DatabaseService


def envelope(via: str, from_agent: str, message: str) -> str:
    """The message with its provenance envelope, as the model sees it."""
    tag = f"[via:{via} from:{from_agent}]" if from_agent and from_agent != via else f"[via:{via}]"
    return f"{tag} {message}"


class IngressService:
    """Deliver messages from other systems into sessions. Implements LifecycleAware."""

    def __init__(
        self,
        assistant_service: AssistantService,
        streaming_service: StreamingService,
        database_service: DatabaseService | None = None,
        socket_server: Any | None = None,
    ) -> None:
        self._assistant = assistant_service
        self._streaming = streaming_service
        self._db = database_service
        self._sio = socket_server
        self._queued: list[dict[str, Any]] = []  # the inbox when there is no database
        self._background: set[asyncio.Task[None]] = set()
        self._started = False

    async def start(self) -> None:
        self._started = True
        logger.info("Ingress service started")

    async def stop(self) -> None:
        self._started = False
        for task in list(self._background):
            task.cancel()
        for task in list(self._background):
            with contextlib.suppress(BaseException):
                await task
        self._background.clear()
        logger.info("Ingress service stopped")

    async def health_check(self) -> dict:
        return {
            "healthy": self._started,
            "running_deliveries": len(self._background),
            "queued": len(self._queued),
        }

    # --- delivery ---------------------------------------------------------------

    async def deliver(
        self,
        *,
        from_agent: str,
        via: str,
        message: str,
        session_id: str | None = None,
        telegram_chat_id: str | None = None,
        severity: str = "info",
        queue_when_unrouted: bool = True,
    ) -> dict[str, Any]:
        """Deliver ``message`` into a session, or queue it when there is none.

        The session is the one named, else the newest one bound to the
        Telegram chat, else the most recently active one. Returns
        ``{"status": "delivered", "session_id", "delivery"}`` with ``delivery``
        ``queued`` (into a live turn) or ``promoted`` (a turn of its own),
        ``{"status": "queued", "inbox_id"}`` when no session exists, or
        ``{"status": "skipped"}`` when ``queue_when_unrouted`` is False.
        """
        target = await self._resolve_session(session_id, telegram_chat_id)
        if target is None:
            if not queue_when_unrouted:
                return {"status": "skipped", "reason": "no_active_session"}
            inbox_id = await self._queue(from_agent, via, message, severity, session_id)
            return {"status": "queued", "inbox_id": inbox_id}

        request = AssistantRequest(
            id=str(uuid.uuid4()),
            session_id=target,
            message_type="steering",
            content=envelope(via, from_agent, message),
        )
        action = await self._streaming.accept_steering(request, has_live_stream=False)
        if action == "promoted":
            self._run_in_background(request)
        logger.info(
            "Ingress message delivered",
            session_id=target,
            via=via,
            sender=from_agent,
            delivery=action,
        )
        return {"status": "delivered", "session_id": target, "delivery": action}

    async def drain(self, session_id: str) -> int:
        """Queue the waiting inbox messages as steering on ``session_id``; returns how many.

        A message leaves the inbox only once it is queued, so nothing is lost
        when the session is not there yet.
        """
        sessions = self._assistant.get_session_store()
        if sessions is None:
            return 0
        delivered = 0
        for item in await self._waiting(session_id):
            request = AssistantRequest(
                id=f"inbox-{item['id']}",
                session_id=session_id,
                message_type="steering",
                content=envelope(item["via"], item["from_agent"], item["message"]),
            )
            try:
                await sessions.queue_steering(session_id, request)
            except ValueError:
                pass  # already queued on a previous attempt: consume it
            except LookupError:
                break  # no such session yet; leave everything waiting
            await self._consume(item)
            delivered += 1
        if delivered:
            logger.info("Inbox drained into session", session_id=session_id, count=delivered)
        return delivered

    # --- pieces -----------------------------------------------------------------

    async def _resolve_session(
        self, session_id: str | None, telegram_chat_id: str | None
    ) -> str | None:
        sessions = self._assistant.get_session_store()
        if sessions is None:
            return None
        if session_id:
            ctx = await sessions.get_context_if_exists_async(session_id)
            return session_id if ctx is not None else None
        if telegram_chat_id:
            bound = await sessions.get_session_id_for_telegram_chat_async(telegram_chat_id)
            if bound:
                return bound
        listed = await sessions.list_sessions(limit=1)
        if listed:
            latest = listed[0].get("session_id")
            return latest if isinstance(latest, str) and latest else None
        return None

    def _run_in_background(self, request: AssistantRequest) -> None:
        task = asyncio.create_task(self._run(request), name=f"ingress-{request.session_id}")
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _run(self, request: AssistantRequest) -> None:
        """Run the promoted turn; events go to the session's Socket.IO room, if any."""
        room = f"session:{request.session_id}"
        try:
            async for event in self._streaming.stream_message(request):
                if self._sio is not None:
                    with contextlib.suppress(Exception):
                        await self._sio.emit(
                            socket_event_name(event), event, room=room, namespace="/assistant"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Ingress turn failed", session_id=request.session_id, error=str(e))

    async def _queue(
        self, from_agent: str, via: str, message: str, severity: str, session_id: str | None
    ) -> str | None:
        context: dict[str, Any] = {"via": via, "injected": True}
        if session_id:
            context["session_id"] = session_id
        if self._db is not None and getattr(self._db, "healthy", False):
            try:
                from assistant_runtime.services.database.repositories import InboxRepository

                async with self._db.session_context() as db_session:
                    row = await InboxRepository(db_session).create(
                        from_agent=from_agent, message=message, severity=severity, context=context
                    )
                    return row.id
            except Exception as e:
                logger.warning("Inbox write failed; keeping the message in memory", error=str(e))
        item_id = str(uuid.uuid4())
        self._queued.append(
            {
                "id": item_id,
                "from_agent": from_agent,
                "via": via,
                "message": message,
                "context": context,
            }
        )
        return item_id

    async def _waiting(self, session_id: str) -> list[dict[str, Any]]:
        """Messages waiting for ``session_id`` or for no session in particular (not removed)."""

        def _fits(context: dict[str, Any] | None) -> bool:
            wanted = (context or {}).get("session_id")
            return wanted in (None, session_id)

        waiting = [dict(item, source="memory") for item in self._queued if _fits(item["context"])]
        if self._db is None or not getattr(self._db, "healthy", False):
            return waiting
        try:
            from assistant_runtime.services.database.repositories import InboxRepository

            async with self._db.session_context() as db_session:
                for row in await InboxRepository(db_session).list_unsurfaced(limit=50):
                    if _fits(row.context):
                        waiting.append(
                            {
                                "id": row.id,
                                "from_agent": row.from_agent,
                                "via": (row.context or {}).get("via", "inbox"),
                                "message": row.message,
                                "context": row.context,
                                "source": "db",
                            }
                        )
        except Exception as e:
            logger.warning("Inbox read failed", error=str(e))
        return waiting

    async def _consume(self, item: dict[str, Any]) -> None:
        """Take a delivered message out of the inbox."""
        if item["source"] == "memory":
            self._queued = [q for q in self._queued if q["id"] != item["id"]]
            return
        try:
            from assistant_runtime.services.database.repositories import InboxRepository

            async with self._db.session_context() as db_session:
                await InboxRepository(db_session).mark_surfaced(item["id"])
        except Exception as e:
            logger.warning("Inbox mark-surfaced failed", inbox_id=item["id"], error=str(e))
