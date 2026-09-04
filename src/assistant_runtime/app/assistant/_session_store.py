"""Session store with tree-structured conversation state."""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy.exc import DisconnectionError, InterfaceError, OperationalError

from assistant_runtime.app.assistant._serialization import (
    MessageRecord,
    SteeringRecord,
    path_records_to_model_history,
)
from assistant_runtime.base.resilience import retry_with_backoff

_DB_RETRYABLE_EXCEPTIONS = (
    OperationalError,
    DisconnectionError,
    InterfaceError,
    ConnectionError,
    TimeoutError,
)

if TYPE_CHECKING:
    from assistant_runtime.app.assistant.models import AssistantRequest
    from assistant_runtime.services.database.interface import DatabaseService


_MAX_MEMORY_SESSIONS = 200

_STALE_HOST_TOOL_OUTPUT = (
    "[Deferred host tool was not completed before the session was reloaded. "
    "The action did not complete.]"
)


def _repair_stale_tool_segments(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Mark tools without output as stale.

    Returns (repaired_segments, list_of_repaired_tool_ids).
    Original segments are not mutated.
    """
    repaired_ids: list[str] = []
    updated = copy.deepcopy(segments)
    for segment in updated:
        if segment.get("kind") != "tool_group":
            continue
        for tool in segment.get("tools", []):
            if not isinstance(tool, dict):
                continue
            if "output" not in tool:
                tool["output"] = _STALE_HOST_TOOL_OUTPUT
                repaired_ids.append(str(tool.get("id", "")))
    return updated, repaired_ids


def _repair_stale_tools_in_context(
    ctx: dict[str, Any],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Scan all assistant messages for tools without output and mark as stale.

    Returns list of (message_id, repaired_segments) for messages that were repaired.
    Mutates the records in ctx["message_index"] in place.
    """
    repaired: list[tuple[str, list[dict[str, Any]]]] = []
    for msg_id, record in ctx["message_index"].items():
        if record.get("role") != "assistant":
            continue
        segments = record.get("segments")
        if not segments:
            continue
        updated_segments, repaired_ids = _repair_stale_tool_segments(segments)
        if repaired_ids:
            record["segments"] = updated_segments
            repaired.append((msg_id, updated_segments))
            logger.debug(
                "[SESSION] Marked stale host tools",
                message_id=msg_id,
                repaired_tool_ids=repaired_ids,
            )
    return repaired


class SessionStore:
    """Session state storage with optional DB persistence."""

    def __init__(
        self,
        database_service: DatabaseService | None = None,
        session_ttl_hours: int = 24,
    ) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._db: DatabaseService | None = database_service
        self._session_ttl_hours = session_ttl_hours
        self._pending_db_loads: dict[str, asyncio.Task[dict[str, Any] | None]] = {}
        self._pending_db_loads_lock = asyncio.Lock()

    def get_context(self, session_id: str) -> dict[str, Any]:
        """Get or create in-memory session context without DB hydration."""
        if session_id in self._sessions:
            self._touch_session(session_id)
            return self._sessions[session_id]
        self._evict_if_needed()
        ctx = self._build_empty_context()
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
            loaded = await self._load_session_from_db_singleflight(session_id)
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

    async def register_user_message(
        self, request: AssistantRequest
    ) -> tuple[dict[str, Any], MessageRecord]:
        """Persist a user-side message send and update the active cached path."""
        if request.is_steering:
            raise ValueError("Steering messages are stored separately from the conversation tree")

        session_id = request.session_id
        existing = await self.get_context_if_exists_async(session_id)
        ctx = existing

        if request.parent_id is None:
            if ctx is None:
                ctx = self.get_context(session_id)
                await self._ensure_session_row(session_id, ctx)
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

        now = datetime.now(UTC)
        record: MessageRecord = {
            "id": request.id,
            "session_id": session_id,
            "parent_id": request.parent_id,
            "role": "user",
            "message_type": request.message_type,
            "content": request.content,
            "segments": None,
            "usage": None,
            "created_at": now,
        }

        if self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import (
                    MessageRepository,
                    SessionRepository,
                )

                session_repo = SessionRepository(db_session)
                row = await session_repo.get(session_id)
                if row is None:
                    await session_repo.create(
                        session_id=session_id,
                        title=None,
                        expires_at=self._expires_at(),
                    )
                repo = MessageRepository(db_session)
                await repo.create(
                    message_id=record["id"],
                    session_id=session_id,
                    parent_id=record["parent_id"],
                    role=record["role"],
                    message_type=record["message_type"],
                    content=record["content"],
                )

        self._add_message_to_context(ctx, record)
        self._refresh_cached_path_for_new_leaf(ctx, record)
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

        now = created_at or datetime.now(UTC)
        record: MessageRecord = {
            "id": message_id,
            "session_id": session_id,
            "parent_id": parent_id,
            "role": "assistant",
            "message_type": "standard",
            "content": content,
            "segments": segments,
            "usage": usage,
            "created_at": now,
        }

        if self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import MessageRepository

                repo = MessageRepository(db_session)
                await repo.create(
                    message_id=record["id"],
                    session_id=session_id,
                    parent_id=record["parent_id"],
                    role=record["role"],
                    message_type=record["message_type"],
                    content=record["content"],
                    segments=record["segments"],
                    usage=record["usage"],
                )

        self._add_message_to_context(ctx, record)
        self._refresh_cached_path_for_new_leaf(ctx, record)
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
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import MessageRepository

                repo = MessageRepository(db_session)
                await repo.update(message_id, content=content, segments=segments, usage=usage)

        return record

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
        now = datetime.now(UTC)
        queued: SteeringRecord = {
            "id": request.id,
            "session_id": session_id,
            "content": request.content,
            "status": status,
            "created_at": now,
            "delivered_at": delivered_at,
        }

        if self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import SteeringRepository

                repo = SteeringRepository(db_session)
                await repo.create(
                    steering_id=queued["id"],
                    session_id=session_id,
                    content=queued["content"],
                    status=queued["status"],
                    delivered_at=queued["delivered_at"],
                )

        self._add_steering_to_context(ctx, queued)
        return queued

    async def list_pending_steering(
        self,
        session_id: str,
    ) -> list[SteeringRecord]:
        """Return pending steering in submission order."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        return [ctx["steering_index"][gid] for gid in ctx["pending_steering_ids"]]

    async def deliver_pending_steering(
        self,
        session_id: str,
    ) -> list[SteeringRecord]:
        """Mark all currently pending steering as delivered."""
        return await self._mark_pending_steering(session_id, status="delivered")

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
                self._coerce_datetime(record.get("delivered_at"))
                or self._coerce_datetime(record.get("created_at"))
                or datetime.now(UTC),
                self._coerce_datetime(record.get("created_at")) or datetime.now(UTC),
                record["id"],
            ),
        )

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
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import SteeringRepository

                repo = SteeringRepository(db_session)
                await repo.mark_status([steering_id], status="promoted", delivered_at=delivered_at)

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

        target_leaf = leaf_id or ctx.get("active_leaf_id") or self._latest_leaf_id(ctx)
        if target_leaf is None:
            return []
        if target_leaf == ctx.get("active_leaf_id") and ctx["cached_path"]:
            return list(ctx["cached_path"])
        if target_leaf not in ctx["message_index"]:
            raise LookupError(f"Message '{target_leaf}' not found")
        return self._resolve_path(ctx, target_leaf)

    async def get_tree_messages(self, session_id: str) -> list[MessageRecord]:
        """Return all messages in a session ordered by created_at."""
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        return sorted(
            ctx["message_index"].values(),
            key=lambda message: message["created_at"],
        )

    async def save_session_state_async(self, session_id: str) -> None:
        """Persist non-message session metadata."""
        ctx = self._sessions.get(session_id)
        if ctx is None or self._db is None:
            return

        async with self._db.session_context() as db_session:
            from assistant_runtime.services.database.repositories import SessionRepository

            repo = SessionRepository(db_session)
            await repo.upsert(
                session_id,
                title=ctx.get("title"),
                turn_number=ctx.get("turn_number", 0),
                working_memory=ctx.get("working_memory"),
                telegram_chat_id=ctx.get("telegram_chat_id"),
                telegram_bound_at=ctx.get("telegram_bound_at"),
                expires_at=self._expires_at(),
            )

    async def get_session_id_for_telegram_chat_async(self, chat_id: str) -> str | None:
        """Resolve the newest active session bound to a Telegram chat id."""
        best_session_id: str | None = None
        best_bound_at: datetime | None = None
        for session_id, ctx in self._sessions.items():
            if ctx.get("telegram_chat_id") != chat_id:
                continue
            bound_at = self._coerce_datetime(ctx.get("telegram_bound_at"))
            if best_session_id is None or self._is_newer_binding(bound_at, best_bound_at):
                best_session_id = session_id
                best_bound_at = bound_at
        if best_session_id is not None:
            return best_session_id

        if self._db is None:
            return None

        async with self._db.session_context() as db_session:
            from assistant_runtime.services.database.repositories import SessionRepository

            repo = SessionRepository(db_session)
            row = await repo.get_latest_by_telegram_chat_id(chat_id)
            return row.id if row is not None else None

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def session_count(self) -> int:
        return len(self._sessions)

    async def delete_session(self, session_id: str) -> None:
        if self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import SessionRepository

                repo = SessionRepository(db_session)
                await repo.delete(session_id)
        self._sessions.pop(session_id, None)

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """List sessions with metadata and tree message counts."""
        if self._db is None:
            sessions = []
            for sid, ctx in self._sessions.items():
                sessions.append(
                    {
                        "session_id": sid,
                        "title": ctx.get("title"),
                        "turn_number": ctx.get("turn_number", 0),
                        "message_count": ctx.get("message_count", 0),
                        "created_at": None,
                    }
                )
            return sessions[offset : offset + limit]

        async with self._db.session_context() as db_session:
            from assistant_runtime.services.database.repositories import (
                MessageRepository,
                SessionRepository,
            )

            session_repo = SessionRepository(db_session)
            message_repo = MessageRepository(db_session)
            rows = await session_repo.list_all(limit=limit, offset=offset)
            counts = await message_repo.count_by_sessions([row.id for row in rows])

            db_results = [
                {
                    "session_id": row.id,
                    "title": row.title,
                    "turn_number": row.turn_number,
                    "message_count": counts.get(row.id, 0),
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

            for result in db_results:
                sid = result["session_id"]
                if sid in self._sessions:
                    ctx = self._sessions[sid]
                    result["message_count"] = ctx.get("message_count", result["message_count"])
                    result["turn_number"] = ctx.get("turn_number", result["turn_number"])
                    if ctx.get("title"):
                        result["title"] = ctx["title"]

            return db_results

    async def cleanup_expired(self) -> int:
        if self._db is None:
            return 0
        async with self._db.session_context() as db_session:
            from assistant_runtime.services.database.repositories import SessionRepository

            repo = SessionRepository(db_session)
            return await repo.cleanup_expired()

    async def repair_stale_host_tools(self, session_id: str) -> dict[str, Any]:
        """Repair stale host tools in a session.

        Clears in-memory pending state and marks any tool without output
        as stale in both the in-memory context and the database.

        Returns a report dict with what was repaired.
        """
        report: dict[str, Any] = {
            "session_id": session_id,
            "cleared_pending": None,
            "repaired_tools": [],
        }

        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError(f"Session '{session_id}' not found")

        # Clear in-memory pending state
        cleared = ctx.get("pending_tool_call_id")
        if cleared:
            report["cleared_pending"] = {
                "tool_call_id": cleared,
                "tool_name": ctx.get("pending_tool_name"),
            }
            ctx.pop("pending_tool_call_id", None)
            ctx.pop("pending_tool_name", None)
            ctx.pop("pending_assistant_message_id", None)

        # Repair stale tool segments
        repaired = _repair_stale_tools_in_context(ctx)
        if repaired and self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import MessageRepository

                repo = MessageRepository(db_session)
                for msg_id, updated_segments in repaired:
                    await repo.update(msg_id, segments=updated_segments)

        for msg_id, _segments in repaired:
            report["repaired_tools"].append(msg_id)

        # Also update cached_path to reflect repaired segments
        if repaired:
            active_leaf = ctx.get("active_leaf_id")
            if active_leaf:
                ctx["cached_path"] = self._resolve_path(ctx, active_leaf)

        if cleared or repaired:
            logger.info(
                "[SESSION] Repaired stale host tools",
                session_id=session_id,
                cleared_pending=bool(cleared),
                repaired_messages=len(repaired),
            )

        return report

    async def _ensure_session_row(self, session_id: str, ctx: dict[str, Any]) -> None:
        if self._db is None:
            return
        async with self._db.session_context() as db_session:
            from assistant_runtime.services.database.repositories import SessionRepository

            repo = SessionRepository(db_session)
            existing = await repo.get(session_id)
            if existing is None:
                await repo.create(
                    session_id=session_id, title=ctx.get("title"), expires_at=self._expires_at()
                )

    def _build_empty_context(self) -> dict[str, Any]:
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

    def _add_message_to_context(self, ctx: dict[str, Any], record: MessageRecord) -> None:
        message_index = ctx["message_index"]
        children_by_parent = ctx["children_by_parent"]
        message_index[record["id"]] = record
        children_by_parent.setdefault(record.get("parent_id"), []).append(record["id"])
        children_by_parent.setdefault(record["id"], [])
        ctx["message_count"] = len(message_index)

    def _add_steering_to_context(self, ctx: dict[str, Any], record: SteeringRecord) -> None:
        ctx["steering_index"][record["id"]] = record
        if record["id"] not in ctx["steering_order"]:
            ctx["steering_order"].append(record["id"])
        if record.get("status") == "pending" and record["id"] not in ctx["pending_steering_ids"]:
            ctx["pending_steering_ids"].append(record["id"])

    def _refresh_cached_path_for_new_leaf(self, ctx: dict[str, Any], record: MessageRecord) -> None:
        active_leaf_id = ctx.get("active_leaf_id")
        if active_leaf_id == record.get("parent_id") or (
            active_leaf_id is None and record.get("parent_id") is None
        ):
            ctx["cached_path"] = [*ctx["cached_path"], record]
        else:
            parent_path = self._resolve_path(ctx, record.get("parent_id"))
            ctx["cached_path"] = [*parent_path, record]
        ctx["active_leaf_id"] = record["id"]

    def _resolve_path(self, ctx: dict[str, Any], leaf_id: str | None) -> list[MessageRecord]:
        if leaf_id is None:
            return []
        index = ctx["message_index"]
        path: list[MessageRecord] = []
        current_id = leaf_id
        while current_id is not None:
            message = index.get(current_id)
            if message is None:
                raise LookupError(f"Message '{current_id}' not found")
            path.append(message)
            current_id = message.get("parent_id")
        path.reverse()
        return path

    def _latest_leaf_id(self, ctx: dict[str, Any]) -> str | None:
        latest: MessageRecord | None = None
        children_by_parent = ctx["children_by_parent"]
        for message in ctx["message_index"].values():
            if children_by_parent.get(message["id"]):
                continue
            if latest is None or message["created_at"] > latest["created_at"]:
                latest = message
        return latest["id"] if latest is not None else None

    def _evict_if_needed(self) -> None:
        if len(self._sessions) <= _MAX_MEMORY_SESSIONS:
            return
        evict_count = len(self._sessions) // 2
        keys_to_evict = list(self._sessions.keys())[:evict_count]
        for key in keys_to_evict:
            del self._sessions[key]
        logger.info(
            "[SESSION] Evicted in-memory sessions",
            evicted=evict_count,
            remaining=len(self._sessions),
        )

    def _touch_session(self, session_id: str) -> None:
        ctx = self._sessions.get(session_id)
        if ctx is None:
            return
        self._sessions.pop(session_id, None)
        self._sessions[session_id] = ctx

    async def _load_session_from_db_singleflight(self, session_id: str) -> dict[str, Any] | None:
        if self._db is None:
            return None

        async with self._pending_db_loads_lock:
            existing = self._pending_db_loads.get(session_id)
            if existing is None:
                task = asyncio.create_task(self._load_session_from_db(session_id))
                self._pending_db_loads[session_id] = task
            else:
                task = existing

        try:
            return await task
        finally:
            async with self._pending_db_loads_lock:
                current = self._pending_db_loads.get(session_id)
                if current is task:
                    self._pending_db_loads.pop(session_id, None)

    async def _load_session_from_db(self, session_id: str) -> dict[str, Any] | None:
        if self._db is None:
            return None

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0.1,
            max_wait=1.0,
            retry_on=_DB_RETRYABLE_EXCEPTIONS,
        )
        async def _load() -> dict[str, Any] | None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import (
                    MessageRepository,
                    SessionRepository,
                    SteeringRepository,
                )

                session_repo = SessionRepository(db_session)
                message_repo = MessageRepository(db_session)
                row = await session_repo.get(session_id)
                if row is None:
                    return None
                messages = await message_repo.list_by_session(session_id)
                steering_records = await SteeringRepository(db_session).list_by_session(session_id)
                ctx = self._build_empty_context()
                ctx["turn_number"] = row.turn_number
                ctx["working_memory"] = row.working_memory
                ctx["title"] = row.title
                ctx["telegram_chat_id"] = row.telegram_chat_id
                ctx["telegram_bound_at"] = row.telegram_bound_at

                for message in messages:
                    record: MessageRecord = {
                        "id": message.id,
                        "session_id": message.session_id,
                        "parent_id": message.parent_id,
                        "role": message.role,
                        "message_type": message.message_type,
                        "content": message.content,
                        "segments": message.segments,
                        "usage": message.usage,
                        "created_at": message.created_at,
                    }
                    self._add_message_to_context(ctx, record)

                for steering in steering_records:
                    record: SteeringRecord = {
                        "id": steering.id,
                        "session_id": steering.session_id,
                        "content": steering.content,
                        "status": steering.status,
                        "created_at": steering.created_at,
                        "delivered_at": steering.delivered_at,
                    }
                    self._add_steering_to_context(ctx, record)

                ctx["active_leaf_id"] = self._latest_leaf_id(ctx)
                ctx["cached_path"] = self._resolve_path(ctx, ctx["active_leaf_id"])

                # Repair stale host tools on cold load.
                # pending_tool_call_id is in-memory only — on reload it's gone,
                # so any host tool without output will never get a continuation.
                repaired = _repair_stale_tools_in_context(ctx)
                if repaired:
                    for msg_id, updated_segments in repaired:
                        await message_repo.update(msg_id, segments=updated_segments)
                    logger.info(
                        "[SESSION] Repaired stale host tools on DB load",
                        session_id=session_id,
                        repaired_messages=len(repaired),
                    )

                return ctx

        return await _load()

    def _expires_at(self) -> datetime:
        return datetime.now(UTC) + timedelta(hours=self._session_ttl_hours)

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value
        return None

    @staticmethod
    def _is_newer_binding(candidate: datetime | None, current: datetime | None) -> bool:
        if candidate is None:
            return False
        if current is None:
            return True
        return candidate > current

    async def _mark_pending_steering(
        self,
        session_id: str,
        *,
        status: str,
    ) -> list[SteeringRecord]:
        ctx = await self.get_context_if_exists_async(session_id)
        if ctx is None:
            raise LookupError("Session not found")
        if not ctx["pending_steering_ids"]:
            return []

        delivered_at = datetime.now(UTC)
        updated: list[SteeringRecord] = []
        for steering_id in list(ctx["pending_steering_ids"]):
            record = dict(ctx["steering_index"][steering_id])
            record["status"] = status
            record["delivered_at"] = delivered_at
            ctx["steering_index"][steering_id] = record
            updated.append(record)

        ctx["pending_steering_ids"] = []

        if self._db is not None:
            async with self._db.session_context() as db_session:
                from assistant_runtime.services.database.repositories import SteeringRepository

                repo = SteeringRepository(db_session)
                await repo.mark_status(
                    [record["id"] for record in updated],
                    status=status,
                    delivered_at=delivered_at,
                )

        return updated
