"""TaskService — work run as turns in the background: tasks and persistent agents.

A task is a turn in its own fresh session (``task-<id>``), started by the
model (the task tools), by a host (``POST /api/tasks``) or in process, and
run outside any conversation turn: the caller keeps serving while it runs.
Tasks about the same (profile, subject) run one at a time, in order; at
most ``max_concurrent`` run at once. When a task ends, its record says how
and ``task_finished`` is published on ``app.state.events``; nothing is
steered into the session that asked, so the host decides when and how a
result is reviewed. A restart marks unfinished tasks ``interrupted`` and
publishes that; nothing is replayed.

A persistent agent is started once by the host and keeps one continuing
session until it is stopped. Each message sent to it (by the model's
``message_agent`` tool or by the host) runs as the next turn in that
session, one at a time, in order, within the same ``max_concurrent``; its
end is published as ``agent_message_finished`` with the sender's session.
Each message's turn sends the agent's ``config`` as a request's config, read
when it starts, as its session is. The host moves an agent to a fresh
session when its conversation should start over. A restart keeps the agents
and marks unfinished messages ``interrupted``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from assistant_runtime.app.assistant.config import AssistantConfig, TunableOverrides
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.app.settings import outside_request_allowance
from assistant_runtime.app.tasks._runner import INTERRUPTED, BackgroundTurnRunner
from assistant_runtime.app.tasks._store import (
    AgentTakenError,
    DatabaseTaskStore,
    InMemoryTaskStore,
    TaskStore,
)
from assistant_runtime.app.tasks._tools import register_task_tools
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.exceptions import (
    AgentConflictError,
    AgentNotFoundError,
    TaskError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.app.tasks.models import AgentMessageRecord, AgentRecord, TaskRecord
from assistant_runtime.principal import LOCAL_PRINCIPAL, Principal

if TYPE_CHECKING:
    from assistant_runtime.app.streaming.interface import StreamingService
    from assistant_runtime.base.cursors import ChangeCursor
    from assistant_runtime.base.events import EventHub
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.tools.interface import ToolService


class TaskService:
    """Start, list, read and cancel background tasks. Implements LifecycleAware."""

    def __init__(
        self,
        config: TasksConfig,
        streaming_service: StreamingService,
        database_service: DatabaseService | None = None,
        events: EventHub | None = None,
        tool_service: ToolService | None = None,
        default_profile: str | None = None,
        assistant_config: AssistantConfig | None = None,
    ) -> None:
        self._config = config
        # Its request allowances decide which agent configs are accepted.
        self._assistant_config = assistant_config or AssistantConfig()
        self._tool_service = tool_service
        self._default_profile = default_profile
        self._streaming = streaming_service
        self._database = database_service
        self._events = events
        self._store: TaskStore | None = None
        self._runner = BackgroundTurnRunner(config, streaming_service)
        # Agent starts, moves, stops and message admission see one another's changes.
        self._agent_admission = asyncio.Lock()

    # --- lifecycle ---

    async def start(self) -> None:
        database = self._database
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseTaskStore(database)
            for record in await self._store.mark_unfinished("interrupted", INTERRUPTED):
                await self._publish(record)
            for message in await self._store.mark_unfinished_messages("interrupted", INTERRUPTED):
                await self._publish_message(message)
            for agent in await self._store.list_agents(created_by=None, status="active"):
                self._streaming.pin_session(agent.session_id)
        else:
            self._store = InMemoryTaskStore()
        if self._config.enabled and self._tool_service is not None:
            # The tool service starts first (registration order), so its registry is ready.
            register_task_tools(self._tool_service, self)
        logger.info("Task service started", enabled=self._config.enabled)

    async def stop(self) -> None:
        await self._runner.stop()
        self._store = None
        logger.info("Task service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._store is not None,
            "enabled": self._config.enabled,
            "running": self._runner.count,
            "durable": self._store.durable if self._store is not None else None,
        }

    # --- public API ---

    async def start_task(
        self,
        task: str,
        *,
        profile: str | None = None,
        subject: str | None = None,
        context: str | None = None,
        parent_session_id: str | None = None,
        principal: Principal | None = None,
    ) -> TaskRecord:
        """Queue a task and return its record at once; it runs in the background."""
        self._require_enabled()
        store = self._require_store()
        task = (task or "").strip()
        if not task:
            raise TaskError("A task needs a description of the work")
        self._check_names(profile, subject)
        principal = principal or LOCAL_PRINCIPAL
        task_id = str(uuid.uuid4())
        with self._runner.reserve():
            record = await store.create(
                TaskRecord(
                    id=task_id,
                    session_id=f"task-{task_id}",
                    task=task,
                    status="queued",
                    created_by=principal.id,
                    # The default resolved now: omitting it and naming it are the same profile.
                    profile=profile or self._default_profile,
                    subject=subject,
                    context=(context or "").strip() or None,
                    parent_session_id=parent_session_id,
                )
            )
            self._runner.submit(
                task_id,
                self._run(record, principal),
                store.update,
                self._publish,
                name=f"task-{task_id}",
            )
        logger.info("Task queued", task_id=task_id, profile=profile, subject=subject)
        return record

    async def get(self, task_id: str, principal: Principal | None = None) -> TaskRecord:
        record = await self._require_store().get(task_id)
        if record is None or not _visible(record, principal or LOCAL_PRINCIPAL):
            raise TaskNotFoundError(f"No task '{task_id}'")
        return record

    async def list(
        self,
        principal: Principal | None = None,
        *,
        parent_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        before: str | None = None,
        updated_after: ChangeCursor | None = None,
    ) -> list[TaskRecord]:
        """Newest first: the caller's own tasks, or every task for an administrator.

        ``before`` names a task the caller can see; only older ones are listed
        (``TaskNotFoundError`` when it is unknown). With ``updated_after``, the
        tasks changed after that cursor, oldest change first.
        """
        principal = principal or LOCAL_PRINCIPAL
        anchor = await self.get(before, principal) if before is not None else None
        return await self._require_store().list(
            created_by=None if principal.is_admin else principal.id,
            parent_session_id=parent_session_id,
            status=status,
            limit=limit,
            before=anchor,
            updated_after=updated_after,
        )

    async def count(self, start: datetime, end: datetime) -> int:
        """Tasks created in ``[start, end)``, everyone's; 0 while the service is not started."""
        return await self._store.count(start, end) if self._store is not None else 0

    async def count_messages(self, start: datetime, end: datetime) -> int:
        """Agent messages created in ``[start, end)``, everyone's; 0 while not started."""
        return await self._store.count_messages(start, end) if self._store is not None else 0

    async def cancel(self, task_id: str, principal: Principal | None = None) -> TaskRecord:
        """Stop a queued or running task; a finished one is returned as it is."""
        record = await self.get(task_id, principal)
        if record.finished:
            return record
        await self._runner.cancel(task_id)
        return await self.get(task_id, principal)

    # --- persistent agents ---

    async def start_agent(
        self,
        *,
        profile: str | None = None,
        subject: str | None = None,
        session_id: str | None = None,
        config: TunableOverrides | None = None,
        principal: Principal | None = None,
    ) -> AgentRecord:
        """Start a persistent agent for (profile, subject); it keeps one session until stopped.

        ``session_id`` names the session it continues (a new one is created
        by its first message); by default it gets ``agent-<id>``. ``config``
        is what its turns send as a request's config.
        """
        self._require_enabled()
        store = self._require_store()
        self._check_names(profile, subject, session_id)
        if config is not None:
            self._check_config(config)
        principal = principal or LOCAL_PRINCIPAL
        agent_id = str(uuid.uuid4())
        record = AgentRecord(
            id=agent_id,
            session_id=session_id or f"agent-{agent_id}",
            status="active",
            created_by=principal.id,
            # The default resolved now, as for tasks: omitting it and naming it are one agent.
            profile=profile or self._default_profile,
            subject=subject,
            config=config.model_dump(exclude_none=True) if config is not None else {},
        )
        async with self._agent_admission:
            await self._check_free(record.session_id, identity=(record.profile, record.subject))
            try:
                record = await store.create_agent(record)
            except AgentTakenError as e:
                raise AgentConflictError(f"The agent's identity or session is taken: {e}") from e
            # Its conversation stays cached until it moves or stops.
            self._streaming.pin_session(record.session_id)
        logger.info("Agent started", agent_id=agent_id, profile=record.profile, subject=subject)
        return record

    async def move_agent(
        self, agent_id: str, *, session_id: str | None = None, principal: Principal | None = None
    ) -> AgentRecord:
        """Move the agent to a fresh session; messages that start from now on run there."""
        store = self._require_store()
        self._check_names(None, None, session_id)
        async with self._agent_admission:
            agent = await self.get_agent(agent_id, principal)
            if agent.status != "active":
                raise AgentConflictError(f"Agent '{agent_id}' is stopped")
            session_id = session_id or f"agent-{agent_id}-{uuid.uuid4().hex[:8]}"
            if session_id == agent.session_id:
                return agent
            await self._check_free(session_id)
            try:
                moved = await store.update_agent(agent_id, session_id=session_id)
            except AgentTakenError as e:
                raise AgentConflictError(f"Session '{session_id}' is taken: {e}") from e
            self._streaming.pin_session(session_id)
            self._streaming.unpin_session(agent.session_id)
        logger.info("Agent moved", agent_id=agent_id, session_id=session_id)
        return moved or agent

    async def configure_agent(
        self, agent_id: str, config: TunableOverrides, *, principal: Principal | None = None
    ) -> AgentRecord:
        """Change the agent's config as ``PATCH /api/settings`` changes the runtime's.

        A field ``config`` leaves unset is unchanged and ``None`` clears one;
        messages that start from now on run with the result.
        """
        store = self._require_store()
        self._check_config(config)
        async with self._agent_admission:
            agent = await self.get_agent(agent_id, principal)
            if agent.status != "active":
                raise AgentConflictError(f"Agent '{agent_id}' is stopped")
            merged = {**agent.config, **config.model_dump(exclude_unset=True)}
            configured = await store.update_agent(
                agent_id, config={k: v for k, v in merged.items() if v is not None}
            )
        logger.info("Agent configured", agent_id=agent_id)
        return configured or agent

    async def stop_agent(self, agent_id: str, principal: Principal | None = None) -> AgentRecord:
        """Stop the agent; its queued and running messages end ``cancelled``."""
        store = self._require_store()
        async with self._agent_admission:
            agent = await self.get_agent(agent_id, principal)
            if agent.status != "active":
                return agent
            stopped = await store.update_agent(agent_id, status="stopped", stopped_at=_now())
        await self._runner.cancel_agent(agent_id)
        self._streaming.unpin_session(agent.session_id)
        logger.info("Agent stopped", agent_id=agent_id)
        return stopped or agent

    async def get_agent(self, agent_id: str, principal: Principal | None = None) -> AgentRecord:
        record = await self._require_store().get_agent(agent_id)
        if record is None or not _visible(record, principal or LOCAL_PRINCIPAL):
            raise AgentNotFoundError(f"No agent '{agent_id}'")
        return record

    async def list_agents(
        self, principal: Principal | None = None, *, status: str | None = None
    ) -> list[AgentRecord]:
        """Newest first: the caller's own agents, or every agent for an administrator."""
        principal = principal or LOCAL_PRINCIPAL
        return await self._require_store().list_agents(
            created_by=None if principal.is_admin else principal.id, status=status
        )

    async def message_agent(
        self,
        content: str,
        *,
        agent_id: str | None = None,
        profile: str | None = None,
        subject: str | None = None,
        parent_session_id: str | None = None,
        principal: Principal | None = None,
    ) -> AgentMessageRecord:
        """Queue a message for an active agent, named by id or by (profile, subject).

        It runs as the agent's next turn, after the messages before it; the
        record is returned at once.
        """
        self._require_enabled()
        store = self._require_store()
        content = (content or "").strip()
        if not content:
            raise TaskError("A message needs content")
        principal = principal or LOCAL_PRINCIPAL
        async with self._agent_admission:
            agent = await self._find_agent(agent_id, profile, subject, principal)
            if agent.status != "active":
                raise AgentConflictError(f"Agent '{agent.id}' is stopped")
            message_id = str(uuid.uuid4())
            with self._runner.reserve():
                record = await store.create_message(
                    AgentMessageRecord(
                        id=message_id,
                        agent_id=agent.id,
                        content=content,
                        status="queued",
                        created_by=principal.id,
                        parent_session_id=parent_session_id,
                    )
                )
                self._runner.submit(
                    message_id,
                    self._run_message(record, principal),
                    store.update_message,
                    self._publish_message,
                    name=f"agent-message-{message_id}",
                    agent_id=agent.id,
                )
        logger.info("Agent message queued", agent_id=agent.id, message_id=message_id)
        return record

    async def get_message(
        self, message_id: str, principal: Principal | None = None
    ) -> AgentMessageRecord:
        record = await self._require_store().get_message(message_id)
        if record is None or not _visible(record, principal or LOCAL_PRINCIPAL):
            raise AgentNotFoundError(f"No agent message '{message_id}'")
        return record

    async def list_messages(
        self,
        principal: Principal | None = None,
        *,
        agent_id: str | None = None,
        parent_session_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        before: str | None = None,
        updated_after: ChangeCursor | None = None,
    ) -> list[AgentMessageRecord]:
        """Newest first: the caller's own messages, or every message for an administrator.

        ``before`` names a message the caller can see (of ``agent_id``, when
        given); only older ones are listed (``AgentNotFoundError`` when it is unknown).
        With ``updated_after``, the messages changed after that cursor, oldest change first.
        """
        principal = principal or LOCAL_PRINCIPAL
        anchor = None
        if before is not None:
            anchor = await self.get_message(before, principal)
            if agent_id is not None and anchor.agent_id != agent_id:
                raise AgentNotFoundError(f"No agent message '{before}'")
        return await self._require_store().list_messages(
            agent_id=agent_id,
            created_by=None if principal.is_admin else principal.id,
            parent_session_id=parent_session_id,
            status=status,
            limit=limit,
            before=anchor,
            updated_after=updated_after,
        )

    # --- running ---

    async def _run(self, record: TaskRecord, principal: Principal) -> None:
        store = self._require_store()

        async def start() -> AssistantRequest | None:
            started = await store.update(
                record.id, only_from=("queued",), status="running", started_at=_now()
            )
            if started is None:
                return None  # it ended before it could start; its end was recorded then
            return AssistantRequest(
                id=str(uuid.uuid4()),
                session_id=record.session_id,
                content=_content(record),
                profile=record.profile,
                subject=record.subject,
            )

        subject_key = (record.profile, record.subject) if record.subject is not None else None
        await self._runner.execute(
            record.id, subject_key, start, store.update, self._publish, principal
        )

    async def _run_message(self, record: AgentMessageRecord, principal: Principal) -> None:
        store = self._require_store()

        async def start() -> AssistantRequest | None:
            # Under admission, so a move is wholly before this start or wholly after it.
            async with self._agent_admission:
                agent = await store.get_agent(record.agent_id)
                if agent is None or agent.status != "active":
                    # Stopped elsewhere while this waited: end it as the stop would have.
                    self._runner.mark_cancelled(record.id)
                    raise asyncio.CancelledError
                # The agent's session now, so a move applies to every message not yet started.
                started = await store.update_message(
                    record.id,
                    only_from=("queued",),
                    status="running",
                    session_id=agent.session_id,
                    started_at=_now(),
                )
            if started is None:
                return None  # it ended before it could start; its end was recorded then
            return AssistantRequest(
                id=str(uuid.uuid4()),
                session_id=agent.session_id,
                content=record.content,
                profile=agent.profile,
                subject=agent.subject,
                # Always an object: a missing config would reuse the session's last one.
                config=TunableOverrides(**agent.config),
            )

        await self._runner.execute(
            record.id,
            ("agent", record.agent_id),
            start,
            store.update_message,
            self._publish_message,
            principal,
        )

    async def _publish(self, record: TaskRecord) -> None:
        if self._events is None:
            return
        await self._events.publish(
            {
                "type": "task_finished",
                "task_id": record.id,
                "status": record.status,
                "session_id": record.session_id,
                "parent_session_id": record.parent_session_id,
                "profile": record.profile,
                "subject": record.subject,
                "result": record.result,
                "error": record.error,
            }
        )

    async def _publish_message(self, record: AgentMessageRecord) -> None:
        if self._events is None:
            return
        await self._events.publish(
            {
                "type": "agent_message_finished",
                "message_id": record.id,
                "agent_id": record.agent_id,
                "status": record.status,
                "session_id": record.session_id,
                "parent_session_id": record.parent_session_id,
                "result": record.result,
                "error": record.error,
            }
        )

    def _require_enabled(self) -> None:
        if not self._config.enabled:
            raise TasksDisabledError("Background tasks are not enabled")

    def _require_store(self) -> TaskStore:
        if self._store is None:
            raise TaskError("Task service not started")
        return self._store

    def _check_names(
        self, profile: str | None, subject: str | None, session_id: str | None = None
    ) -> None:
        """The request's own rules for profile, subject and session names, checked up front."""
        self._streaming.validate_profile(profile)
        try:
            AssistantRequest(
                id="check",
                session_id=session_id or "check",
                content="check",
                profile=profile,
                subject=subject,
            )
        except ValidationError as e:
            names = "profile, subject or session" if session_id else "profile or subject"
            raise TaskError(f"Invalid {names}: {e.errors()[0]['msg']}") from e

    def _check_config(self, config: TunableOverrides) -> None:
        """Refuse a value a turn would ignore; the allowances are fixed at startup."""
        refused = [
            f"{field}={value!r}"
            for field, value in config.model_dump(exclude_none=True).items()
            if outside_request_allowance(self._assistant_config, field, value)
        ]
        if refused:
            raise TaskError(
                "Not allowed in a request's config (ASSISTANT__REQUEST_MODELS, "
                f"ASSISTANT__REQUEST_SERVICE_TIER): {', '.join(refused)}"
            )

    async def _check_free(
        self, session_id: str, *, identity: tuple[str | None, str | None] | None = None
    ) -> None:
        """Refuse a session, or an identity (profile, subject) when given, already in use.

        Call it holding ``_agent_admission``.
        """
        for agent in await self._require_store().list_agents(created_by=None, status="active"):
            if identity is not None and (agent.profile, agent.subject) == identity:
                raise AgentConflictError(
                    f"Agent '{agent.id}' is already active for this profile and subject"
                )
            if agent.session_id == session_id:
                raise AgentConflictError(f"Agent '{agent.id}' already uses session '{session_id}'")

    async def _find_agent(
        self,
        agent_id: str | None,
        profile: str | None,
        subject: str | None,
        principal: Principal,
    ) -> AgentRecord:
        if agent_id is not None:
            return await self.get_agent(agent_id, principal)
        profile = profile or self._default_profile
        for agent in await self.list_agents(principal, status="active"):
            if (agent.profile, agent.subject) == (profile, subject):
                return agent
        raise AgentNotFoundError(
            f"No active agent for profile '{profile}' and subject '{subject or ''}'"
        )


def _visible(record: TaskRecord | AgentRecord | AgentMessageRecord, principal: Principal) -> bool:
    return principal.is_admin or record.created_by == principal.id


def _content(record: TaskRecord) -> str:
    if not record.context:
        return record.task
    return f"{record.task}\n\nContext:\n{record.context}"


def _now() -> datetime:
    return datetime.now(UTC)
