"""AssistantService — core request handler that orchestrates LLM + history + tools.

Public facade for the assistant module. Implements LifecycleAware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.result import DeferredToolRequests

from lovely_assistant.app.assistant._prompt_builder import build_system_prompt
from lovely_assistant.app.assistant._session_store import SessionStore
from lovely_assistant.app.assistant.config import AssistantConfig
from lovely_assistant.app.assistant.exceptions import AgentRunError, AssistantError, SessionError
from lovely_assistant.app.assistant.models import AssistantRequest, AssistantResult
from lovely_assistant.services.tools.models import DeferredToolRequest

if TYPE_CHECKING:
    from lovely_assistant.services.history.interface import HistoryService
    from lovely_assistant.services.llm.interface import LlmService
    from lovely_assistant.services.tools.interface import ToolService


class AssistantService:
    """Core assistant orchestrator. Implements LifecycleAware."""

    def __init__(
        self,
        config: AssistantConfig,
        llm_service: LlmService,
        history_service: HistoryService,
        tool_service: ToolService,
    ) -> None:
        self._config = config
        self._llm = llm_service
        self._history = history_service
        self._tools = tool_service
        self._sessions: SessionStore | None = None
        self._started = False

    async def start(self) -> None:
        """Initialize internal components."""
        self._sessions = SessionStore()
        self._started = True
        logger.info("Assistant service started")

    async def stop(self) -> None:
        """Shutdown the assistant service."""
        self._sessions = None
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

    async def process_message(self, request: AssistantRequest) -> AssistantResult:
        """Process a user message through the full assistant pipeline.

        Orchestrates: tools → prompt → agent → history → result.

        Args:
            request: The assistant request containing message, session, and context.

        Returns:
            AssistantResult with text content or deferred tool request.

        Raises:
            AssistantError: If the service is not started.
            AgentRunError: If agent execution fails.
            SessionError: If continuation state is invalid.
        """
        self._ensure_started()

        is_continuation = request.tool_call_id is not None

        if is_continuation:
            return await self._handle_continuation(request)
        return await self._handle_new_message(request)

    async def _handle_new_message(self, request: AssistantRequest) -> AssistantResult:
        """Handle a fresh user message (not a continuation)."""
        session_id = request.session_id
        sessions = self._sessions
        assert sessions is not None

        # 1. Get session context and increment turn
        session_context = sessions.get_context(session_id)
        turn_number = sessions.increment_turn(session_id)

        # 2. Get available tools for this request's machine state
        available_tools = self._tools.get_available_tools(request.machine_state)
        toolsets = self._tools.build_toolset(request.machine_state)

        # 3. Build system prompt from fragments
        system_prompt = build_system_prompt(
            available_tools=available_tools,
            session_context=session_context,
            machine_state=request.machine_state,
        )

        # 4. Determine output type based on frontend tools
        has_frontend_tools = len(available_tools.frontend_tools) > 0
        output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

        # 5. Create per-request agent
        model = self._config.default_model
        agent = self._llm.build_agent(
            system_prompt=system_prompt,
            toolsets=toolsets,
            model=model,
            output_type=output_type,
            thinking_budget=self._config.thinking_budget,
        )

        # 6. Prepare history
        history = sessions.get_history(session_id)
        prepared_history, _context_modified = await self._history.prepare_history(
            history, session_context
        )

        # 7. Run the agent
        try:
            result = await agent.run(
                request.message,
                message_history=prepared_history if prepared_history else None,
            )
        except Exception as e:
            raise AgentRunError(f"Agent execution failed: {e}") from e

        # 8. Save message history
        sessions.save_history(session_id, list(result.all_messages()))

        # 9. Handle output
        output = result.output
        resolved_model = model or self._llm._config.primary_model

        if isinstance(output, DeferredToolRequests):
            return self._handle_deferred_output(output, session_id, turn_number, resolved_model)

        # 10. Extract working memory delta (fire-and-forget style)
        if self._config.enable_working_memory:
            await self._update_working_memory(session_id, session_context, turn_number)

        return AssistantResult(
            content=str(output),
            model=resolved_model,
            session_id=session_id,
            turn_number=turn_number,
        )

    async def _handle_continuation(self, request: AssistantRequest) -> AssistantResult:
        """Handle a continuation request (frontend returning tool result)."""
        sessions = self._sessions
        assert sessions is not None
        session_id = request.session_id

        if not sessions.has_session(session_id):
            raise SessionError(f"No session found for continuation: {session_id}")

        # Validate pending tool call
        pending = sessions.clear_pending_tool_call(session_id)
        if pending is None:
            raise SessionError(f"No pending tool call for session {session_id}")

        session_context = sessions.get_context(session_id)
        turn_number = sessions.increment_turn(session_id)

        # Build deferred tool results
        assert request.tool_call_id is not None
        deferred_results = DeferredToolResults(calls={request.tool_call_id: request.tool_result})

        # Rebuild agent with same configuration
        available_tools = self._tools.get_available_tools(request.machine_state)
        toolsets = self._tools.build_toolset(request.machine_state)

        system_prompt = build_system_prompt(
            available_tools=available_tools,
            session_context=session_context,
            machine_state=request.machine_state,
        )

        has_frontend_tools = len(available_tools.frontend_tools) > 0
        output_type: Any = [str, DeferredToolRequests] if has_frontend_tools else str

        model = self._config.default_model
        agent = self._llm.build_agent(
            system_prompt=system_prompt,
            toolsets=toolsets,
            model=model,
            output_type=output_type,
            thinking_budget=self._config.thinking_budget,
        )

        # Get saved history and prepare (with is_continuation=True)
        history = sessions.get_history(session_id)
        prepared_history, _ = await self._history.prepare_history(
            history, session_context, is_continuation=True
        )

        # Run with deferred tool results
        try:
            result = await agent.run(
                None,
                message_history=prepared_history if prepared_history else None,
                deferred_tool_results=deferred_results,
            )
        except Exception as e:
            raise AgentRunError(f"Continuation failed: {e}") from e

        # Save updated history
        sessions.save_history(session_id, list(result.all_messages()))

        output = result.output
        resolved_model = model or self._llm._config.primary_model

        if isinstance(output, DeferredToolRequests):
            return self._handle_deferred_output(output, session_id, turn_number, resolved_model)

        if self._config.enable_working_memory:
            await self._update_working_memory(session_id, session_context, turn_number)

        return AssistantResult(
            content=str(output),
            model=resolved_model,
            session_id=session_id,
            turn_number=turn_number,
        )

    def _handle_deferred_output(
        self,
        deferred: DeferredToolRequests,
        session_id: str,
        turn_number: int,
        model: str,
    ) -> AssistantResult:
        """Convert DeferredToolRequests into an AssistantResult with a DeferredToolRequest."""
        sessions = self._sessions
        assert sessions is not None

        calls: list[ToolCallPart] = deferred.calls
        if not calls:
            raise AgentRunError("DeferredToolRequests has no calls")

        # Take the first call (single-tool enforcement)
        call = calls[0]
        tool_call_id = call.tool_call_id

        # Store pending state for continuation
        sessions.set_pending_tool_call(session_id, tool_call_id, call.tool_name)

        # Build args dict
        args: dict[str, Any] = {}
        if isinstance(call.args, dict):
            args = call.args
        elif isinstance(call.args, str):
            import json

            try:
                args = json.loads(call.args)
            except (json.JSONDecodeError, TypeError):
                args = {}

        return AssistantResult(
            content=None,
            model=model,
            deferred_tool_request=DeferredToolRequest(
                request_id=tool_call_id,
                tool_name=call.tool_name,
                arguments=args,
            ),
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
            # Get recent messages as simple dicts for analysis
            sessions = self._sessions
            assert sessions is not None
            history = sessions.get_history(session_id)
            recent = [{"role": "message", "index": i} for i, _ in enumerate(history[-4:])]

            updated_wm = await self._history.extract_memory_delta(current_wm, recent, turn_number)
            session_context["working_memory"] = updated_wm
        except Exception as e:
            # Working memory extraction is best-effort — don't fail the request
            logger.warning("Working memory extraction failed", error=str(e))

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._sessions is None:
            raise AssistantError("Assistant service not started")
