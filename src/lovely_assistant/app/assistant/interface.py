"""AssistantService — core request handler that orchestrates LLM + history + tools.

Public facade for the assistant module. Implements LifecycleAware.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolRequests
from pydantic_ai.usage import UsageLimits

from lovely_assistant.app.assistant._prompt_builder import build_system_prompt
from lovely_assistant.app.assistant._serialization import build_assistant_message_content
from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.exceptions import (
    AgentRunError,
    AssistantError,
)
from lovely_assistant.app.assistant.models import (
    AgentSetupContext,
    AssistantRequest,
    AssistantResult,
    _build_user_prompt,
)
from lovely_assistant.app.settings import RuntimeSettings, resolve_effective_config
from lovely_assistant.services.tools._request_context import (
    assistant_request_context,
    get_current_telegram_chat_binding,
)
from lovely_assistant.services.tools._screen_tools import (
    clear_current_screenshot,
    extract_screenshot_data_uri,
    set_current_screenshot,
)
from lovely_assistant.services.tracing import create_request_trace, create_span

if TYPE_CHECKING:
    from lovely_assistant.services.database.interface import DatabaseService
    from lovely_assistant.services.history.interface import HistoryService
    from lovely_assistant.services.llm.interface import LlmService
    from lovely_assistant.services.tools.interface import ToolService

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
        self._sessions = SessionStore(
            database_service=self._database_service,
            session_ttl_hours=self._config.session_ttl_hours,
        )
        self._started = True
        logger.info("Assistant service started")

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
        machine_state: dict[str, Any] | None = None,
    ) -> None:
        """Warm session-local and shared request-path caches.

        This keeps the first user message from paying cold session hydration and
        shared prompt-input discovery when the dashboard has already joined the
        session.
        """
        self._ensure_started()
        sessions = self._sessions
        assert sessions is not None

        started_at = time.monotonic()
        await sessions.get_context_async(session_id)
        if machine_state is not None:
            sessions.get_context(session_id)["last_machine_state"] = machine_state
        try:
            self._tools.warm_machine_state(machine_state)
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
            page=machine_state.get("active_page", {}).get("name")
            if isinstance(machine_state, dict)
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
            machine_state = (
                request.machine_state
                if request.machine_state is not None
                else session_context.get("last_machine_state")
            )
            request_config = request.config or session_context.get("last_request_config")
            if request.machine_state is not None:
                session_context["last_machine_state"] = request.machine_state
            if request.config is not None:
                session_context["last_request_config"] = request.config

            # 1. Tools
            available_tools = self._tools.get_available_tools(machine_state)
            toolsets = self._tools.build_toolset(machine_state)

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
            if artifacts is None:
                raise AssistantError("Cannot build system prompt: artifact store unavailable")

            prompt_result = build_system_prompt(
                available_tools=available_tools,
                session_context=session_context,
                machine_state=machine_state,
                mcp_summary=mcp_summary,
                artifacts=artifacts,
            )

            # 4. Build agent — use union output type when frontend tools are registered
            output_type: type | list[type] = str
            if available_tools.frontend_tools:
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

    async def process_message(self, request: AssistantRequest) -> AssistantResult:
        """Process a user message through the full assistant pipeline.

        Orchestrates: tools → prompt → agent → history → result.

        Args:
            request: The assistant request containing message, session, and context.

        Returns:
            AssistantResult with text content.

        Raises:
            AssistantError: If the service is not started.
            AgentRunError: If agent execution fails.
        """
        self._ensure_started()

        started_at = time.monotonic()
        session_id = request.session_id
        sessions = self._sessions
        assert sessions is not None

        # 1. Persist the user message and cache the active leaf path
        session_context, _user_record = await sessions.register_user_message(request)
        turn_number = session_context.get("turn_number", 0)
        assistant_message_id = str(uuid.uuid4())
        session_context["current_assistant_message_id"] = assistant_message_id

        # 2. Build agent setup context (tools → MCP → artifacts → prompt → config → agent)
        ctx = await self.prepare_agent_context(request, session_context)

        logger.info(
            "Executing assistant request",
            session_id=session_id,
            turn_number=turn_number,
            model=ctx.resolved_model,
            has_images=bool(request.images),
            max_turns=ctx.effective_config.max_turns,
            thinking_budget=ctx.effective_config.thinking_budget,
            temperature=ctx.effective_config.temperature,
        )

        with create_request_trace(
            session_id=session_id,
            model=ctx.resolved_model,
            is_continuation=False,
            input_message=request.content,
        ) as trace:
            # 3. Prepare history
            history = sessions.get_history(session_id, exclude_leaf=True)
            with create_span("history-preparation"):
                prepared_history, _context_modified = await self._history.prepare_history(
                    history, session_context
                )

            # 4. Run the agent
            user_prompt = _build_user_prompt(request.content)
            screenshot = extract_screenshot_data_uri(
                images=request.images,
                tool_result=request.tool_result,
            )
            if screenshot:
                set_current_screenshot(screenshot)
            telegram_chat_id: str | None = None
            try:
                with assistant_request_context(session_id):
                    result = await ctx.agent.run(
                        user_prompt,
                        message_history=prepared_history if prepared_history else None,
                        usage_limits=ctx.usage_limits,
                    )
                    telegram_chat_id = get_current_telegram_chat_binding()
            except Exception as e:
                logger.exception(
                    "Assistant request failed",
                    session_id=session_id,
                    model=ctx.resolved_model,
                    duration_ms=(time.monotonic() - started_at) * 1000,
                    error=str(e),
                )
                raise AgentRunError(f"Agent execution failed: {e}") from e
            finally:
                clear_current_screenshot()

            if telegram_chat_id:
                session_context["telegram_chat_id"] = telegram_chat_id
                session_context["telegram_bound_at"] = datetime.now(UTC)
                await sessions.save_session_state_async(session_id)

            # 5. Persist the assistant message row for this turn
            # Use new_messages() — not index slicing — because Pydantic AI's
            # _clean_message_history may merge consecutive ModelRequests (e.g.
            # a SYNTHETIC ToolReturn + user prompt), shrinking the list and
            # making len(prepared_history) overshoot.
            turn_messages = list(result.new_messages())
            assistant_content, assistant_segments, assistant_timestamp = (
                build_assistant_message_content(turn_messages)
            )
            usage = self._safe_usage(result)
            await sessions.register_assistant_message(
                session_id,
                message_id=assistant_message_id,
                parent_id=request.id,
                content=assistant_content,
                segments=assistant_segments,
                usage=usage,
                created_at=assistant_timestamp,
            )
            session_context.pop("current_assistant_message_id", None)

            # 6. Handle output
            output = result.output
            logger.info(
                "Assistant request completed",
                session_id=session_id,
                model=ctx.resolved_model,
                duration_ms=(time.monotonic() - started_at) * 1000,
                output_type=type(output).__name__,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
            )

            trace.update_output(str(output))

        # 7. Extract working memory delta (fire-and-forget style)
        if ctx.effective_config.enable_working_memory:
            await self._update_working_memory(session_id, session_context, turn_number)
            await sessions.save_session_state_async(session_id)

        return AssistantResult(
            content=str(output),
            model=ctx.resolved_model,
            session_id=session_id,
            turn_number=turn_number,
        )

    async def _update_working_memory(
        self,
        session_id: str,
        session_context: dict[str, Any],
        turn_number: int,
    ) -> None:
        """Extract working memory delta from recent messages (best-effort)."""
        try:
            from lovely_assistant.services.history.models import WorkingMemory

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
        except Exception as e:
            # Working memory extraction is best-effort — don't fail the request
            logger.warning("Working memory extraction failed", error=str(e))

    async def _load_active_artifacts(self) -> dict[str, str] | None:
        """Load all active artifacts from DB. Returns None if DB unavailable."""
        if self._database_service is None:
            return None
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

                from lovely_assistant.services.database.repositories import ArtifactRepository

                async with self._database_service.session_context() as session:
                    repo = ArtifactRepository(session)
                    rows = await repo.get_all_active()
                    artifacts = {row.name: row.content for row in rows}
                    self._active_artifacts_cache = artifacts
                    self._active_artifacts_cached_at = time.monotonic()
                    return dict(artifacts)
        except Exception as e:
            logger.warning("Failed to load artifacts from DB", error=str(e))
            return None

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._sessions is None:
            raise AssistantError("Assistant service not started")

    @staticmethod
    def _safe_usage(result: Any) -> dict[str, int | None]:
        """Extract usage stats from a run result (best-effort)."""
        try:
            usage = result.usage()
            return {
                "input_tokens": usage.request_tokens,
                "output_tokens": usage.response_tokens,
                "total_tokens": usage.total_tokens,
            }
        except Exception as e:
            logger.debug("Failed to extract usage stats", error=str(e))
            return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
