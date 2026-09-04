"""AssistantService — sessions, prompt artifacts and per-request agent setup.

The turn itself (running the agent, persisting the reply, streaming events)
lives in ``app/streaming``; this service owns what a turn is built from.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from assistant_runtime.app.assistant._defaults import load_default_artifacts
from assistant_runtime.app.assistant._prompt_builder import build_system_prompt
from assistant_runtime.app.assistant._session_store import SessionStore
from assistant_runtime.app.assistant.config import AssistantConfig
from assistant_runtime.app.assistant.exceptions import AssistantError
from assistant_runtime.app.assistant.models import AgentSetupContext, AssistantRequest
from assistant_runtime.app.settings import RuntimeSettings, resolve_effective_config
from assistant_runtime.services.tracing import create_span

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService
    from assistant_runtime.services.history.interface import HistoryService
    from assistant_runtime.services.llm.interface import LlmService
    from assistant_runtime.services.tools.interface import ToolService

_ARTIFACT_CACHE_TTL_SECONDS = 5.0


class AssistantService:
    """Core assistant orchestrator. Implements LifecycleAware."""

    def __init__(
        self,
        config: AssistantConfig,
        llm_service: LlmService,
        history_service: HistoryService,
        tool_service: ToolService,
        runtime_settings: RuntimeSettings | None = None,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._history = history_service
        self._tools = tool_service
        self._runtime_settings = runtime_settings
        self._database_service: DatabaseService | None = database_service
        self._sessions: SessionStore | None = None
        self._active_artifacts_cache: dict[str, str] | None = None
        self._active_artifacts_cached_at = 0.0
        self._active_artifacts_lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        """Initialize internal components."""
        db = self._usable_database()
        self._sessions = SessionStore(
            database_service=db,
            session_ttl_hours=self._config.session_ttl_hours,
        )
        self._started = True
        logger.info("Assistant service started", persistence="database" if db else "memory")

    async def stop(self) -> None:
        """Shutdown the assistant service."""
        self._sessions = None
        self._active_artifacts_cache = None
        self._active_artifacts_cached_at = 0.0
        self._started = False
        logger.info("Assistant service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        if not self._started or self._sessions is None:
            return {"healthy": False}
        return {
            "healthy": True,
            "active_sessions": self._sessions.session_count(),
        }

    async def cleanup_expired_sessions(self) -> int:
        """Delete expired sessions from DB. Returns count deleted."""
        if self._sessions is None:
            return 0
        return await self._sessions.cleanup_expired()

    def set_runtime_settings(self, runtime_settings: RuntimeSettings | None) -> None:
        """Attach live runtime settings after service construction."""
        self._runtime_settings = runtime_settings

    def get_session_store(self) -> SessionStore | None:
        """Return the session store, or None if service not started."""
        return self._sessions

    def get_database_service(self) -> DatabaseService | None:
        """Return the database service for route-level access."""
        return self._database_service

    async def warm_session(
        self,
        session_id: str,
        host_context: dict[str, Any] | None = None,
    ) -> None:
        """Warm session-local and shared request-path caches.

        This keeps the first user message from paying cold session hydration and
        shared prompt-input discovery when the host has already joined the
        session.
        """
        self._ensure_started()
        sessions = self._sessions
        assert sessions is not None

        started_at = time.monotonic()
        await sessions.get_context_async(session_id)
        if host_context is not None:
            sessions.get_context(session_id)["last_host_context"] = host_context
        try:
            self._tools.warm_host_context(host_context)
        except Exception as e:
            logger.warning(
                "Session warmup step failed",
                session_id=session_id,
                step="tool_registry",
                error=str(e),
            )

        warm_results = await asyncio.gather(
            self._tools.get_mcp_summary(),
            self._load_active_artifacts(),
            return_exceptions=True,
        )

        for label, result in zip(
            ("mcp_summary", "active_artifacts"),
            warm_results,
            strict=False,
        ):
            if isinstance(result, Exception):
                logger.warning(
                    "Session warmup step failed",
                    session_id=session_id,
                    step=label,
                    error=str(result),
                )

        logger.debug(
            "Session warmup completed",
            session_id=session_id,
            page=host_context.get("page", {}).get("name")
            if isinstance(host_context, dict)
            else None,
            duration_ms=(time.monotonic() - started_at) * 1000,
        )

    async def prepare_agent_context(
        self,
        request: AssistantRequest,
        session_context: dict[str, Any],
    ) -> AgentSetupContext:
        """Build the full agent setup context for a request.

        Shared by both AssistantService and StreamingService to prevent drift.
        """
        with create_span("agent-setup"):
            host_context = (
                request.host_context
                if request.host_context is not None
                else session_context.get("last_host_context")
            )
            request_config = request.config or session_context.get("last_request_config")
            if request.host_context is not None:
                session_context["last_host_context"] = request.host_context
            if request.config is not None:
                session_context["last_request_config"] = request.config

            # 1. Tools
            available_tools = self._tools.get_available_tools(host_context)
            toolsets = self._tools.build_toolset(host_context)

            mcp_summary_task = asyncio.create_task(self._tools.get_mcp_summary())
            artifacts_task = asyncio.create_task(self._load_active_artifacts())

            # 2. Config resolution can run while prompt inputs load.
            effective = resolve_effective_config(
                self._config, self._runtime_settings, request_config
            )
            resolved_model = self._llm.resolve_model(effective.default_model)
            usage_limits = UsageLimits(request_limit=effective.max_turns)

            # 3. MCP + artifacts + system prompt
            mcp_summary, artifacts = await asyncio.gather(mcp_summary_task, artifacts_task)

            prompt_result = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                host_context=host_context,
                mcp_summary=mcp_summary,
                artifacts=artifacts,
            )

            # 4. Build agent — use union output type when host tools are registered
            output_type: type | list[type] = str
            if available_tools.host_tools:
                output_type = [str, DeferredToolRequests]

            agent = self._llm.build_agent(
                system_prompt=prompt_result.content,
                toolsets=toolsets,
                model=resolved_model,
                output_type=output_type,
                thinking_budget=effective.thinking_budget,
                temperature=effective.temperature,
            )

        return AgentSetupContext(
            agent=agent,
            available_tools=available_tools,
            toolsets=toolsets,
            prompt_result=prompt_result,
            resolved_model=resolved_model,
            usage_limits=usage_limits,
            output_type=output_type,
            effective_config=effective,
            mcp_summary=mcp_summary,
        )

    async def update_working_memory(
        self,
        session_id: str,
        session_context: dict[str, Any],
        turn_number: int,
    ) -> None:
        """Extract the working-memory delta from the recent path and persist it (best-effort)."""
        try:
            from assistant_runtime.services.history.models import WorkingMemory

            current_wm = session_context.get("working_memory") or WorkingMemory()
            sessions = self._sessions
            assert sessions is not None
            path = await sessions.get_message_path(session_id)

            recent: list[dict[str, Any]] = []
            for message in path[-4:]:
                if message.get("role") not in {"user", "assistant"}:
                    continue
                recent.append(
                    {
                        "role": message["role"],
                        "content": message.get("content", ""),
                    }
                )

            updated_wm = await self._history.extract_memory_delta(current_wm, recent, turn_number)
            session_context["working_memory"] = updated_wm
            await sessions.save_session_state_async(session_id)
        except Exception as e:
            # Working memory extraction is best-effort — don't fail the request
            logger.warning("Working memory extraction failed", error=str(e))

    def _usable_database(self) -> DatabaseService | None:
        """The database service when it is reachable, else None (memory-only mode)."""
        db = self._database_service
        if db is None or not getattr(db, "healthy", False):
            return None
        return db

    async def _load_active_artifacts(self) -> dict[str, str]:
        """Active prompt artifacts: bundled defaults overlaid with the database rows.

        Without a reachable database, or for any name the database has no
        active row for, the bundled default text is used, so the runtime can
        build a system prompt with nothing configured beyond a provider key.
        """
        defaults = load_default_artifacts()
        if self._usable_database() is None:
            return defaults
        now = time.monotonic()
        if (
            self._active_artifacts_cache is not None
            and (now - self._active_artifacts_cached_at) < _ARTIFACT_CACHE_TTL_SECONDS
        ):
            return dict(self._active_artifacts_cache)
        try:
            async with self._active_artifacts_lock:
                now = time.monotonic()
                if (
                    self._active_artifacts_cache is not None
                    and (now - self._active_artifacts_cached_at) < _ARTIFACT_CACHE_TTL_SECONDS
                ):
                    return dict(self._active_artifacts_cache)

                from assistant_runtime.services.database.repositories import ArtifactRepository

                async with self._database_service.session_context() as session:
                    repo = ArtifactRepository(session)
                    rows = await repo.get_all_active()
                    artifacts = {**defaults, **{row.name: row.content for row in rows}}
                    self._active_artifacts_cache = artifacts
                    self._active_artifacts_cached_at = time.monotonic()
                    return dict(artifacts)
        except Exception as e:
            logger.warning("Failed to load artifacts from DB, using defaults", error=str(e))
            return defaults

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._sessions is None:
            raise AssistantError("Assistant service not started")
