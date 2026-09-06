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

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic_ai import DeferredToolResults
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
)

from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.assistant._serialization import (
    assistant_record_to_flat_messages,
    path_records_to_model_history,
)
from assistant_runtime.app.assistant._stale_tools import find_tool_entry
from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.assistant.models import (
    AssistantRequest,
    extract_screenshot_data_uri,
    strip_screenshot_from_tool_result,
)
from assistant_runtime.app.streaming._host_tool import clear_stale_pending_call
from assistant_runtime.app.streaming.exceptions import StreamSetupError
from assistant_runtime.principal import LOCAL_PRINCIPAL, Principal, can_access_session

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore

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
    # Who runs the turn; set by ``plan()`` and available to tools through the request context.
    principal: Principal = LOCAL_PRINCIPAL
    # Model messages already persisted on the assistant row being extended.
    prior_assistant_messages: list[ModelMessage] = field(default_factory=list)
    prior_usage: dict[str, int] | None = None
    # What the agent run starts from.
    user_prompt: str | Sequence[UserContent] | None = None
    history: list[ModelMessage] = field(default_factory=list)
    deferred_tool_results: DeferredToolResults | None = None
    accepted_tool_result: ModelRequest | None = None
    suppress_tool_call_ids: set[str] = field(default_factory=set)
    # A continuation for one call of a batch with more calls waiting: record the
    # result, hand ``next_pending`` to the host, and do not run the model yet.
    # ``pending_batch`` keeps the model's call order; the set above is for lookups.
    next_pending: dict[str, Any] | None = None
    pending_batch: list[str] = field(default_factory=list)
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

    async def plan(
        self, request: AssistantRequest, principal: Principal = LOCAL_PRINCIPAL
    ) -> TurnPlan:
        """Validate ``request`` against its session and describe the run.

        ``principal`` owns a session it creates and must be allowed on one
        that exists (``AccessDeniedError`` otherwise).
        """
        try:
            if request.is_steering:
                plan = await self._plan_steering(request, principal)
            elif request.is_continuation:
                plan = await self._plan_continuation(request, principal)
            else:
                plan = await self._plan_message(request, principal)
            plan.principal = principal
            return plan
        except (SessionError, AccessDeniedError):
            raise
        except (LookupError, ValueError) as e:
            # The session store rejects requests that do not fit the tree
            # (unknown session or parent, duplicate id): the client's mistake.
            raise SessionError(str(e)) from e
        except Exception as e:
            raise StreamSetupError(f"Turn setup failed: {e}") from e

    def _authorize(self, session_context: dict[str, Any], principal: Principal, session_id: str):
        if not can_access_session(principal, session_context.get("owner_id")):
            raise AccessDeniedError(f"Session '{session_id}' belongs to another principal")

    async def _plan_message(self, request: AssistantRequest, principal: Principal) -> TurnPlan:
        session_id = request.session_id
        existing = await self._sessions.get_context_if_exists_async(session_id)
        if existing is not None:
            self._authorize(existing, principal, session_id)
            # Resolve a pending host action as superseded *before* the new user
            # message is written: a crash in between must not leave a restored
            # pending action next to the message that superseded it.
            await clear_stale_pending_call(self._sessions, session_id, existing)
        session_context, _user_record = await self._sessions.register_user_message(
            request, owner_id=principal.id
        )
        # Registration may have hydrated a stored context that this call did not see.
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
            user_prompt=build_user_prompt(request),
            history=self._sessions.get_history(session_id, exclude_leaf=True),
            screenshot=request.screenshot,
            turn_number=session_context.get("turn_number", 0),
            input_message=request.content,
            trace_metadata={
                "has_images": bool(request.images),
                "attachments": len(request.attachments),
            },
        )

    async def _plan_continuation(self, request: AssistantRequest, principal: Principal) -> TurnPlan:
        session_id = request.session_id
        session_context = await self._sessions.get_context_if_exists_async(session_id)
        if session_context is None:
            raise SessionError(f"Continuation rejected: session '{session_id}' does not exist")
        self._authorize(session_context, principal, session_id)

        recorded = _recorded_call(session_context, request.tool_call_id or "")
        if recorded is not None and "output" in recorded:
            # The result of this call is already on the assistant row (a
            # duplicate continuation, or a retry after the first was accepted).
            # It is never applied twice; the conversation continues with a new
            # message instead.
            raise SessionError(
                f"Continuation rejected: the result for tool call '{request.tool_call_id}' "
                f"was already recorded (status: {recorded.get('status', 'completed')}); "
                "send a new message to continue"
            )
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
        screenshot = request.screenshot or extract_screenshot_data_uri(
            tool_result=request.tool_result
        )
        tool_result = (
            strip_screenshot_from_tool_result(request.tool_result)
            if screenshot
            else request.tool_result
        )
        pending_tool_name = session_context.get("pending_tool_name") or "unknown"
        batch = [
            str(c) for c in (session_context.get("pending_tool_batch") or [pending_tool_call_id])
        ]
        remaining = [
            call_id
            for call_id in batch
            if call_id != pending_tool_call_id
            and (entry := find_tool_entry(assistant_record.get("segments"), call_id)) is not None
            and "output" not in entry
        ]
        next_pending = None
        if remaining:
            entry = find_tool_entry(assistant_record.get("segments"), remaining[0]) or {}
            next_pending = {
                "tool_name": str(entry.get("name", "")),
                "call_id": remaining[0],
                "arguments": entry.get("input") if isinstance(entry.get("input"), dict) else {},
                "queued": remaining[1:],
            }
        if request.tool_outcome == "failed":
            # A host-declared failure is the native ``failed`` outcome: the model
            # sees the failure message and does not repeat the call.
            failure = _failure_message(tool_result)
            deferred_result: Any = ToolFailed(failure)
            accepted_content: Any = failure
        else:
            deferred_result = tool_result
            accepted_content = tool_result
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
            deferred_tool_results=(
                None
                if next_pending
                else DeferredToolResults(calls={request.tool_call_id: deferred_result})
            ),
            next_pending=next_pending,
            pending_batch=batch,
            accepted_tool_result=ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name=pending_tool_name,
                        tool_call_id=pending_tool_call_id,
                        content=accepted_content,
                        outcome="failed" if request.tool_outcome == "failed" else "success",
                    )
                ]
            ),
            suppress_tool_call_ids=set(batch),
            screenshot=screenshot,
            turn_number=session_context.get("turn_number", 0),
            update_working_memory=False,
            input_message=request.content or "(continuation)",
        )

    async def _plan_steering(self, request: AssistantRequest, principal: Principal) -> TurnPlan:
        session_id = request.session_id
        session_context = await self._sessions.get_context_if_exists_async(session_id)
        if session_context is None:
            raise SessionError(f"Steering rejected: session '{session_id}' does not exist")
        self._authorize(session_context, principal, session_id)
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
            await self._sessions.queue_steering(session_id, request)
        elif steering_record.get("status") == "delivered":
            raise SessionError(f"Steering '{request.id}' was already delivered by another turn")

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
            turn_number=session_context.get("turn_number", 0),
            input_message=request.content,
            trace_metadata={"steering": True},
        )


def _recorded_call(session_context: dict[str, Any], tool_call_id: str) -> dict[str, Any] | None:
    """The stored tool entry for ``tool_call_id`` on any assistant row of the session.

    The pending and active rows are checked first; an abandoned call can sit
    on an older row once the conversation has moved on.
    """
    if not tool_call_id:
        return None
    index = session_context["message_index"]
    first = [
        session_context.get("pending_assistant_message_id"),
        session_context.get("active_leaf_id"),
    ]
    ordered = [index[m] for m in first if m in index] + [
        record for record in index.values() if record["id"] not in first
    ]
    for record in ordered:
        if record.get("role") != "assistant":
            continue
        entry = find_tool_entry(record.get("segments"), tool_call_id)
        if entry is not None:
            return entry
    return None


def _failure_message(tool_result: Any) -> str:
    """The host's failure result as the text the model reads."""
    if tool_result is None:
        return "The host reported that the action failed."
    if isinstance(tool_result, str):
        return tool_result
    return json.dumps(tool_result, default=str)


def build_user_prompt(request: AssistantRequest) -> str | list[UserContent]:
    """The user prompt with reference attachments as native Pydantic AI content.

    Screenshots are not included: they stay available to ``look_at_screen``.
    """
    parts: list[UserContent] = []
    for attachment in request.reference_attachments:
        if attachment.text is not None:
            label = attachment.name or "attachment"
            parts.append(f"[{label}]\n{attachment.text}")
        elif attachment.data_uri is not None:
            parts.append(BinaryContent.from_data_uri(attachment.data_uri))
        elif attachment.url is not None:
            parts.append(
                ImageUrl(url=attachment.url)
                if attachment.kind == "image"
                else DocumentUrl(url=attachment.url)
            )
    if not parts:
        return request.content
    return [request.content, *parts] if request.content else parts
