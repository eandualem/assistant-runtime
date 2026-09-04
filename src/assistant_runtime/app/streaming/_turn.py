"""Turn planning: what one agent run needs, for each of the three request kinds.

A turn is one of:

- a **message** — a new user message; the assistant reply becomes a new
  message row under it;
- a **continuation** — the host returns the result of a deferred host-tool
  call; the run resumes and the existing assistant row is extended;
- a **promoted steering** — a steering message that arrived while the session
  was idle; it runs immediately, extending the last assistant row or creating
  one when the leaf is a user message.

The planner validates the request against the session and produces a
``TurnPlan``; the runner (``_runner.py``) executes plans without knowing
which kind it has. Validation failures raise ``SessionError`` (the client's
fault, not retryable) and anything else ``StreamSetupError``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart

from assistant_runtime.app.assistant._serialization import (
    assistant_record_to_flat_messages,
    build_steering_request,
    path_records_to_model_history,
)
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.streaming._host_tool import clear_stale_pending_call
from assistant_runtime.app.streaming.exceptions import StreamSetupError
from assistant_runtime.services.tools._screen_tools import (
    extract_screenshot_data_uri,
    strip_screenshot_from_tool_result,
)

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.app.assistant.models import AssistantRequest

TurnKind = Literal["message", "continuation", "steering"]


@dataclass
class TurnPlan:
    """Everything the runner needs to execute one turn."""

    kind: TurnKind
    request: AssistantRequest
    session_id: str
    session_context: dict[str, Any]
    assistant_message_id: str
    # Set when the turn creates a new assistant row (its parent); None updates an existing row.
    assistant_parent_id: str | None
    # Model messages already persisted on the assistant row being extended.
    prior_assistant_messages: list[ModelMessage] = field(default_factory=list)
    prior_usage: dict[str, int] | None = None
    # What the agent run starts from.
    user_prompt: str | None = None
    history: list[ModelMessage] = field(default_factory=list)
    history_is_continuation: bool = False
    exclude_tool_call_ids: set[str] | None = None
    history_suffix: list[ModelMessage] = field(default_factory=list)
    deferred_tool_results: DeferredToolResults | None = None
    suppress_tool_call_ids: set[str] = field(default_factory=set)
    screenshot: str | None = None
    # Bookkeeping.
    turn_number: int = 0
    update_working_memory: bool = True
    input_message: str = ""
    trace_metadata: dict[str, Any] = field(default_factory=dict)


class TurnPlanner:
    """Turn requests into ``TurnPlan``s against the session store."""

    def __init__(self, sessions: SessionStore) -> None:
        self._sessions = sessions

    async def plan(self, request: AssistantRequest) -> TurnPlan:
        """Validate ``request`` against its session and describe the run."""
        try:
            if request.is_steering:
                return await self._plan_steering(request)
            if request.is_continuation:
                return await self._plan_continuation(request)
            return await self._plan_message(request)
        except SessionError:
            raise
        except (LookupError, ValueError) as e:
            # The session store rejects requests that do not fit the tree
            # (unknown session or parent, duplicate id): the client's mistake.
            raise SessionError(str(e)) from e
        except Exception as e:
            raise StreamSetupError(f"Turn setup failed: {e}") from e

    async def _plan_message(self, request: AssistantRequest) -> TurnPlan:
        session_id = request.session_id
        session_context, _user_record = await self._sessions.register_user_message(request)
        await clear_stale_pending_call(self._sessions, session_id, session_context)
        assistant_message_id = str(uuid.uuid4())
        session_context["current_assistant_message_id"] = assistant_message_id
        return TurnPlan(
            kind="message",
            request=request,
            session_id=session_id,
            session_context=session_context,
            assistant_message_id=assistant_message_id,
            assistant_parent_id=request.id,
            user_prompt=request.content,
            history=self._sessions.get_history(session_id, exclude_leaf=True),
            screenshot=extract_screenshot_data_uri(images=request.images),
            turn_number=session_context.get("turn_number", 0),
            input_message=request.content,
            trace_metadata={"has_images": bool(request.images)},
        )

    async def _plan_continuation(self, request: AssistantRequest) -> TurnPlan:
        session_id = request.session_id
        session_context = await self._sessions.get_context_if_exists_async(session_id)
        if session_context is None:
            raise SessionError(f"Continuation rejected: session '{session_id}' does not exist")

        pending_tool_call_id = session_context.get("pending_tool_call_id")
        if not pending_tool_call_id:
            # A new message cleared the pending state while the host tool ran.
            logger.info(
                "Late continuation arrived after pending state cleared",
                session_id=session_id,
                tool_call_id=request.tool_call_id,
            )
            raise SessionError(
                f"Continuation ignored: session '{session_id}' no longer has a pending "
                "tool call (a new message was processed). The host tool result "
                "arrived after the session moved on."
            )
        if pending_tool_call_id != request.tool_call_id:
            raise SessionError(
                "Continuation rejected: "
                f"tool_call_id '{request.tool_call_id}' does not match pending "
                f"tool call '{pending_tool_call_id}' for session '{session_id}'"
            )
        assistant_message_id = session_context.get("pending_assistant_message_id")
        if not assistant_message_id:
            raise SessionError(
                "Continuation rejected: "
                f"session '{session_id}' is missing the pending assistant message id"
            )
        assistant_record = session_context["message_index"].get(assistant_message_id)
        if assistant_record is None:
            raise SessionError(
                "Continuation rejected: "
                f"assistant message '{assistant_message_id}' is not cached for session '{session_id}'"
            )
        session_context["current_assistant_message_id"] = assistant_message_id

        # pydantic-ai resumes a deferred call from the *last* ModelResponse, so
        # the assistant row is re-serialised flat: completed tools first, the
        # pending call alone at the end.
        flat_assistant = assistant_record_to_flat_messages(assistant_record)
        if not flat_assistant:
            pending_tool_name = session_context.get("pending_tool_name") or "unknown"
            logger.warning(
                "Empty segments for assistant record — synthesizing ModelResponse from pending tool state",
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                pending_tool_call_id=pending_tool_call_id,
                pending_tool_name=pending_tool_name,
            )
            flat_assistant = [
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=pending_tool_name, args={}, tool_call_id=pending_tool_call_id
                        )
                    ],
                    timestamp=datetime.now(UTC),
                )
            ]

        # The host may attach a post-action screenshot; it goes to look_at_screen,
        # not into the model context.
        screenshot = extract_screenshot_data_uri(
            images=request.images, tool_result=request.tool_result
        )
        tool_result = (
            strip_screenshot_from_tool_result(request.tool_result)
            if screenshot
            else request.tool_result
        )
        return TurnPlan(
            kind="continuation",
            request=request,
            session_id=session_id,
            session_context=session_context,
            assistant_message_id=assistant_message_id,
            assistant_parent_id=None,
            prior_assistant_messages=path_records_to_model_history([assistant_record]),
            prior_usage=assistant_record.get("usage"),
            history=[
                *self._sessions.get_history(session_id, exclude_leaf=True),
                *flat_assistant,
            ],
            history_is_continuation=True,
            exclude_tool_call_ids={pending_tool_call_id},
            deferred_tool_results=DeferredToolResults(calls={request.tool_call_id: tool_result}),
            suppress_tool_call_ids={request.tool_call_id},
            screenshot=screenshot,
            turn_number=session_context.get("turn_number", 0),
            update_working_memory=False,
            input_message=request.content or "(continuation)",
        )

    async def _plan_steering(self, request: AssistantRequest) -> TurnPlan:
        session_id = request.session_id
        session_context = await self._sessions.get_context_if_exists_async(session_id)
        if session_context is None:
            raise SessionError(f"Steering rejected: session '{session_id}' does not exist")
        active_leaf_id = session_context.get("active_leaf_id")
        active_leaf = (
            session_context["message_index"].get(active_leaf_id) if active_leaf_id else None
        )
        if active_leaf is None:
            raise SessionError(
                f"Steering rejected: session '{session_id}' has no active conversation"
            )
        if active_leaf.get("role") not in {"user", "assistant"}:
            raise SessionError(
                f"Steering rejected: session '{session_id}' has no active assistant context"
            )

        steering_record = session_context["steering_index"].get(request.id)
        if steering_record is None:
            steering_record = await self._sessions.queue_steering(
                session_id, request, status="promoted", delivered_at=datetime.now(UTC)
            )
        elif steering_record.get("status") == "pending":
            steering_record = await self._sessions.mark_steering_promoted(session_id, request.id)

        extends_assistant = active_leaf.get("role") == "assistant"
        assistant_message_id = active_leaf_id if extends_assistant else str(uuid.uuid4())
        session_context["current_assistant_message_id"] = assistant_message_id
        return TurnPlan(
            kind="steering",
            request=request,
            session_id=session_id,
            session_context=session_context,
            assistant_message_id=assistant_message_id,
            assistant_parent_id=None if extends_assistant else active_leaf_id,
            prior_assistant_messages=(
                path_records_to_model_history([active_leaf]) if extends_assistant else []
            ),
            history=self._sessions.get_history(session_id),
            history_suffix=[build_steering_request([steering_record])],
            turn_number=session_context.get("turn_number", 0),
            input_message=request.content,
            trace_metadata={"steering": True},
        )
