"""Sessions as trees of messages, cached in memory and persisted when possible.

A session context is a dict: the message index and parent/child links, the
cached root-to-leaf path of the active leaf, the steering records and their
pending queue, and per-session state (turn number, working memory, host
context, Telegram binding, the pending host-tool call). The store keeps the
most recent ``_MAX_MEMORY_SESSIONS`` contexts and hydrates others from the
database on demand; every change is written through ``SessionPersistence``
when a database is available.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.assistant._serialization import (
    MessageRecord,
    SteeringRecord,
    path_records_to_model_history,
)
from assistant_runtime.app.assistant._session_persistence import (
    LoadedSession,
    SessionPersistence,
)
from assistant_runtime.app.assistant._stale_tools import repair_stale_tools_in_context

if TYPE_CHECKING:
    from assistant_runtime.app.assistant.models import AssistantRequest
    from assistant_runtime.services.database.interface import DatabaseService


_MAX_MEMORY_SESSIONS = 200


class SessionStore:
    """Session state storage with optional database persistence."""

    def __init__(
        self,
        database_service: DatabaseService | None = None,
        session_ttl_hours: int = 24,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._db: SessionPersistence | None = (
            SessionPersistence(database_service, session_ttl_hours)
            if database_service is not None
            else None
        )
        self._pending_db_loads: dict[str, asyncio.Task[dict[str, Any] | None]] = {}
        self._pending_db_loads_lock = asyncio.Lock()

    # --- contexts -----------------------------------------------------------

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get or create in-memory session context without DB hydration."""
        if session_id in self._sessions:
            self._touch_session(session_id)
            return self._sessions[session_id]
        self._evict_if_needed()
        ctx = _empty_context()
        self._sessions[session_id] = ctx
        return ctx

    async def get_context_async(self, session_id: str) -> dict[str, Any]:
        """Get or create session context, hydrating from DB on cache miss."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is not None:
            return ctx
        return self.get_context(session_id)

    async def get_context_if_exists_async(self, session_id: str) -> dict[str, Any] | None:
        """Return context if the session exists in memory or DB."""
        if session_id in self._sessions:
            self._touch_session(session_id)
            return self._sessions[session_id]

        if self._db is not None:
            loaded = await self._load_session_singleflight(session_id)
            if loaded is not None:
                self._evict_if_needed()
                self._sessions[session_id] = loaded
                self._touch_session(session_id)
                return loaded

        return None

    def get_history(self, session_id: str, *, exclude_leaf: bool = False) -> list[Any]:
        """Return the cached root-to-leaf history as ModelMessages."""
        ctx = self.get_context(session_id)
        path = (
            ctx["cached_path"][:-1] if exclude_leaf and ctx["cached_path"] else ctx["cached_path"]
        )
        return path_records_to_model_history(path)

    @property
    def persistent(self) -> bool:
        """Whether changes are written to a database (else memory only)."""
        return self._db is not None

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def session_count(self) -> int:
        return len(self._sessions)

    # --- messages -----------------------------------------------------------

    async def register_user_message(
        self, request: AssistantRequest
    ) -> tuple[dict[str, Any], MessageRecord]:
        """Persist a user-side message send and update the active cached path."""
        if request.is_steering:
            raise ValueError("Steering messages are stored separately from the conversation tree")

        session_id = request.session_id
        ctx = await self.get_context_if_exists_async(session_id)

        if request.parent_id is None:
            if ctx is None:
                ctx = self.get_context(session_id)
                if self._db is not None:
                    await self._db.ensure_session(session_id, ctx.get("title"))
            elif ctx["message_count"] > 0:
                raise ValueError("Only the first message in a session may have parent_id = null")
        else:
            if ctx is None:
                raise LookupError("Session not found")
            if request.parent_id not in ctx["message_index"]:
                raise LookupError(f"Parent message '{request.parent_id}' not found")

        assert ctx is not None
        if request.id in ctx["message_index"]:
            raise ValueError(f"Message '{request.id}' already exists")

        record: MessageRecord = {
            "id": request.id,
            "session_id": session_id,
            "parent_id": request.parent_id,
            "role": "user",
            "message_type": request.message_type,
            "content": request.content,
            "segments": None,
            "usage": None,
            "created_at": datetime.now(UTC),
        }
        if self._db is not None:
            await self._db.ensure_session(session_id, ctx.get("title"))
            await self._db.create_message(record)

        _add_message(ctx, record)
        _refresh_cached_path_for_new_leaf(ctx, record)
        ctx["turn_number"] = ctx.get("turn_number", 0) + 1
        if not ctx.get("title") and request.message_type == "standard" and request.content.strip():
            text = request.content.strip()
            ctx["title"] = text[:50] + ("..." if len(text) > 50 else "")

        await self.save_session_state_async(session_id)
        return ctx, record

    async def register_assistant_message(
        self,
        session_id: str,
        *,
        message_id: str,
        parent_id: str,
        content: str,
        segments: list[dict[str, Any]] | None,
        usage: dict[str, Any] | None,
        created_at: datetime | None = None,
    ) -> MessageRecord:
        """Persist a single assistant message row and update cache state."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        if parent_id not in ctx["message_index"]:
            raise LookupError(f"Parent message '{parent_id}' not found")
        if ctx["message_index"][parent_id]["role"] != "user":
            raise ValueError("Assistant messages must parent a user message")
        if message_id in ctx["message_index"]:
            raise ValueError(f"Message '{message_id}' already exists")

        record: MessageRecord = {
            "id": message_id,
            "session_id": session_id,
            "parent_id": parent_id,
            "role": "assistant",
            "message_type": "standard",
            "content": content,
            "segments": segments,
            "usage": usage,
            "created_at": created_at or datetime.now(UTC),
        }
        if self._db is not None:
            await self._db.create_message(record)

        _add_message(ctx, record)
        _refresh_cached_path_for_new_leaf(ctx, record)
        return record

    async def update_message(
        self,
        session_id: str,
        message_id: str,
        *,
        content: str | None = None,
        segments: list[dict[str, Any]] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> MessageRecord:
        """Update an existing message row in-memory and in DB."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None or message_id not in ctx["message_index"]:
            raise LookupError(f"Message '{message_id}' not found")
        record = dict(ctx["message_index"][message_id])
        if content is not None:
            record["content"] = content
        if segments is not None:
            record["segments"] = segments
        if usage is not None:
            record["usage"] = usage
        ctx["message_index"][message_id] = record
        ctx["cached_path"] = [
            record if message["id"] == message_id else message for message in ctx["cached_path"]
        ]
        if self._db is not None:
            await self._db.update_message(
                message_id, content=content, segments=segments, usage=usage
            )
        return record

    async def get_message_path(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
    ) -> list[MessageRecord]:
        """Resolve a root-to-leaf path for the requested leaf (or active/latest leaf)."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")

        target_leaf = leaf_id or ctx.get("active_leaf_id") or _latest_leaf_id(ctx)
        if target_leaf is None:
            return []
        if target_leaf == ctx.get("active_leaf_id") and ctx["cached_path"]:
            return list(ctx["cached_path"])
        if target_leaf not in ctx["message_index"]:
            raise LookupError(f"Message '{target_leaf}' not found")
        return _resolve_path(ctx, target_leaf)

    async def get_tree_messages(self, session_id: str) -> list[MessageRecord]:
        """Return all messages in a session ordered by created_at."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        return sorted(ctx["message_index"].values(), key=lambda message: message["created_at"])

    # --- steering -----------------------------------------------------------

    async def queue_steering(
        self,
        session_id: str,
        request: AssistantRequest,
        *,
        status: str = "pending",
        delivered_at: datetime | None = None,
    ) -> SteeringRecord:
        """Persist steering outside the message tree."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        if request.id in ctx["steering_index"]:
            raise ValueError(f"Steering '{request.id}' already exists")
        queued: SteeringRecord = {
            "id": request.id,
            "session_id": session_id,
            "content": request.content,
            "status": status,
            "created_at": datetime.now(UTC),
            "delivered_at": delivered_at,
        }
        if self._db is not None:
            await self._db.create_steering(queued)
        _add_steering(ctx, queued)
        return queued

    async def list_pending_steering(self, session_id: str) -> list[SteeringRecord]:
        """Return pending steering in submission order."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        return [ctx["steering_index"][gid] for gid in ctx["pending_steering_ids"]]

    async def deliver_pending_steering(self, session_id: str) -> list[SteeringRecord]:
        """Mark all currently pending steering as delivered."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        if not ctx["pending_steering_ids"]:
            return []

        delivered_at = datetime.now(UTC)
        updated: list[SteeringRecord] = []
        for steering_id in list(ctx["pending_steering_ids"]):
            record = dict(ctx["steering_index"][steering_id])
            record["status"] = "delivered"
            record["delivered_at"] = delivered_at
            ctx["steering_index"][steering_id] = record
            updated.append(record)
        ctx["pending_steering_ids"] = []

        if self._db is not None:
            await self._db.mark_steering(
                [record["id"] for record in updated], status="delivered", delivered_at=delivered_at
            )
        return updated

    async def mark_steering_promoted(self, session_id: str, steering_id: str) -> SteeringRecord:
        """Mark a queued steering record as promoted for immediate idle delivery."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        if steering_id not in ctx["steering_index"]:
            raise LookupError(f"Steering '{steering_id}' not found")

        delivered_at = datetime.now(UTC)
        record = dict(ctx["steering_index"][steering_id])
        record["status"] = "promoted"
        record["delivered_at"] = delivered_at
        ctx["steering_index"][steering_id] = record
        ctx["pending_steering_ids"] = [
            gid for gid in ctx["pending_steering_ids"] if gid != steering_id
        ]
        if self._db is not None:
            await self._db.mark_steering(
                [steering_id], status="promoted", delivered_at=delivered_at
            )
        return record

    async def get_display_steering(self, session_id: str) -> list[SteeringRecord]:
        """Return delivered/promoted steering ordered for display."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")

        steering = [
            record
            for record in (ctx["steering_index"][gid] for gid in ctx["steering_order"])
            if record.get("status") in {"delivered", "promoted"}
        ]
        return sorted(
            steering,
            key=lambda record: (
                _as_datetime(record.get("delivered_at"))
                or _as_datetime(record.get("created_at"))
                or datetime.now(UTC),
                _as_datetime(record.get("created_at")) or datetime.now(UTC),
                record["id"],
            ),
        )

    # --- session state ------------------------------------------------------

    async def save_session_state_async(self, session_id: str) -> None:
        """Persist non-message session metadata."""
        ctx = self._sessions.get(session_id)
        if ctx is None or self._db is None:
            return
        await self._db.save_state(session_id, ctx)

    async def get_session_id_for_telegram_chat_async(self, chat_id: str) -> str | None:
        """Resolve the newest active session bound to a Telegram chat id."""
        best_session_id: str | None = None
        best_bound_at: datetime | None = None
        for session_id, ctx in self._sessions.items():
            if ctx.get("telegram_chat_id") != chat_id:
                continue
            bound_at = _as_datetime(ctx.get("telegram_bound_at"))
            if best_session_id is None or _is_newer(bound_at, best_bound_at):
                best_session_id = session_id
                best_bound_at = bound_at
        if best_session_id is not None:
            return best_session_id
        if self._db is None:
            return None
        return await self._db.session_for_telegram_chat(chat_id)

    async def delete_session(self, session_id: str) -> None:
        if self._db is not None:
            await self._db.delete(session_id)
        self._sessions.pop(session_id, None)

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List sessions with metadata and tree message counts."""
        if self._db is None:
            sessions = [
                {
                    "session_id": sid,
                    "title": ctx.get("title"),
                    "turn_number": ctx.get("turn_number", 0),
                    "message_count": ctx.get("message_count", 0),
                    "created_at": None,
                }
                for sid, ctx in self._sessions.items()
            ]
            return sessions[offset : offset + limit]

        results = await self._db.list_sessions(limit, offset)
        # The in-memory context is ahead of the row for sessions being worked on.
        for result in results:
            ctx = self._sessions.get(result["session_id"])
            if ctx is None:
                continue
            result["message_count"] = ctx.get("message_count", result["message_count"])
            result["turn_number"] = ctx.get("turn_number", result["turn_number"])
            if ctx.get("title"):
                result["title"] = ctx["title"]
        return results

    async def cleanup_expired(self) -> int:
        if self._db is None:
            return 0
        return await self._db.cleanup_expired()

    async def repair_stale_host_tools(self, session_id: str) -> dict[str, Any]:
        """Clear the pending host-tool call and mark tools without output stale.

        Returns a report of what was repaired.
        """
        report: dict[str, Any] = {
            "session_id": session_id,
            "cleared_pending": None,
            "repaired_tools": [],
        }
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError(f"Session '{session_id}' not found")

        cleared = ctx.get("pending_tool_call_id")
        if cleared:
            report["cleared_pending"] = {
                "tool_call_id": cleared,
                "tool_name": ctx.get("pending_tool_name"),
            }
            ctx.pop("pending_tool_call_id", None)
            ctx.pop("pending_tool_name", None)
            ctx.pop("pending_assistant_message_id", None)

        repaired = repair_stale_tools_in_context(ctx)
        if repaired and self._db is not None:
            await self._db.update_segments(repaired)
        report["repaired_tools"] = [msg_id for msg_id, _segments in repaired]
        if repaired and ctx.get("active_leaf_id"):
            ctx["cached_path"] = _resolve_path(ctx, ctx["active_leaf_id"])

        if cleared or repaired:
            logger.info(
                "[SESSION] Repaired stale host tools",
                session_id=session_id,
                cleared_pending=bool(cleared),
                repaired_messages=len(repaired),
            )
        return report

    # --- cache internals ----------------------------------------------------

    def _evict_if_needed(self) -> None:
        if len(self._sessions) <= _MAX_MEMORY_SESSIONS:
            return
        evict_count = len(self._sessions) // 2
        for key in list(self._sessions.keys())[:evict_count]:
            del self._sessions[key]
        logger.info(
            "[SESSION] Evicted in-memory sessions",
            evicted=evict_count,
            remaining=len(self._sessions),
        )

    def _touch_session(self, session_id: str) -> None:
        """Move the session to the most-recently-used end."""
        ctx = self._sessions.pop(session_id, None)
        if ctx is not None:
            self._sessions[session_id] = ctx

    async def _load_session_singleflight(self, session_id: str) -> dict[str, Any] | None:
        """Hydrate from the database, sharing one load between concurrent callers."""
        async with self._pending_db_loads_lock:
            task = self._pending_db_loads.get(session_id)
            if task is None:
                task = asyncio.create_task(self._load_session(session_id))
                self._pending_db_loads[session_id] = task
        try:
            return await task
        finally:
            async with self._pending_db_loads_lock:
                if self._pending_db_loads.get(session_id) is task:
                    self._pending_db_loads.pop(session_id, None)

    async def _load_session(self, session_id: str) -> dict[str, Any] | None:
        assert self._db is not None
        loaded = await self._db.load(session_id)
        if loaded is None:
            return None
        ctx = _context_from_loaded(loaded)
        # pending_tool_call_id is in-memory only, so after a reload no host
        # tool without output can ever get its continuation: mark them stale.
        repaired = repair_stale_tools_in_context(ctx)
        if repaired:
            await self._db.update_segments(repaired)
            logger.info(
                "[SESSION] Repaired stale host tools on DB load",
                session_id=session_id,
                repaired_messages=len(repaired),
            )
        return ctx


# --- context helpers (pure functions over the context dict) -------------------


def _empty_context() -> dict[str, Any]:
    return {
        "turn_number": 0,
        "working_memory": None,
        "title": None,
        "telegram_chat_id": None,
        "telegram_bound_at": None,
        "last_host_context": None,
        "last_request_config": None,
        "pending_tool_call_id": None,
        "pending_tool_name": None,
        "pending_assistant_message_id": None,
        "current_assistant_message_id": None,
        "pending_steering_ids": [],
        "steering_index": {},
        "steering_order": [],
        "message_count": 0,
        "message_index": {},
        "children_by_parent": {},
        "cached_path": [],
        "active_leaf_id": None,
    }


def _context_from_loaded(loaded: LoadedSession) -> dict[str, Any]:
    ctx = _empty_context()
    ctx["turn_number"] = loaded.turn_number
    ctx["working_memory"] = loaded.working_memory
    ctx["title"] = loaded.title
    ctx["telegram_chat_id"] = loaded.telegram_chat_id
    ctx["telegram_bound_at"] = loaded.telegram_bound_at
    for record in loaded.messages:
        _add_message(ctx, record)
    for steering in loaded.steering:
        _add_steering(ctx, steering)
    ctx["active_leaf_id"] = _latest_leaf_id(ctx)
    ctx["cached_path"] = _resolve_path(ctx, ctx["active_leaf_id"])
    return ctx


def _add_message(ctx: dict[str, Any], record: MessageRecord) -> None:
    message_index = ctx["message_index"]
    children_by_parent = ctx["children_by_parent"]
    message_index[record["id"]] = record
    children_by_parent.setdefault(record.get("parent_id"), []).append(record["id"])
    children_by_parent.setdefault(record["id"], [])
    ctx["message_count"] = len(message_index)


def _add_steering(ctx: dict[str, Any], record: SteeringRecord) -> None:
    ctx["steering_index"][record["id"]] = record
    if record["id"] not in ctx["steering_order"]:
        ctx["steering_order"].append(record["id"])
    if record.get("status") == "pending" and record["id"] not in ctx["pending_steering_ids"]:
        ctx["pending_steering_ids"].append(record["id"])


def _refresh_cached_path_for_new_leaf(ctx: dict[str, Any], record: MessageRecord) -> None:
    active_leaf_id = ctx.get("active_leaf_id")
    if active_leaf_id == record.get("parent_id") or (
        active_leaf_id is None and record.get("parent_id") is None
    ):
        ctx["cached_path"] = [*ctx["cached_path"], record]
    else:
        ctx["cached_path"] = [*_resolve_path(ctx, record.get("parent_id")), record]
    ctx["active_leaf_id"] = record["id"]


def _resolve_path(ctx: dict[str, Any], leaf_id: str | None) -> list[MessageRecord]:
    """Root-to-leaf records for ``leaf_id``."""
    if leaf_id is None:
        return []
    index = ctx["message_index"]
    path: list[MessageRecord] = []
    current_id: str | None = leaf_id
    while current_id is not None:
        message = index.get(current_id)
        if message is None:
            raise LookupError(f"Message '{current_id}' not found")
        path.append(message)
        current_id = message.get("parent_id")
    path.reverse()
    return path


def _latest_leaf_id(ctx: dict[str, Any]) -> str | None:
    """The most recently created message without children."""
    latest: MessageRecord | None = None
    children_by_parent = ctx["children_by_parent"]
    for message in ctx["message_index"].values():
        if children_by_parent.get(message["id"]):
            continue
        if latest is None or message["created_at"] > latest["created_at"]:
            latest = message
    return latest["id"] if latest is not None else None


def _as_datetime(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


def _is_newer(candidate: datetime | None, current: datetime | None) -> bool:
    if candidate is None:
        return False
    return current is None or candidate > current
