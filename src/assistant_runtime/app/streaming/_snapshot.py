"""Native run snapshots, assistant-row persistence and interrupted tool results."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from assistant_runtime.app.assistant import (
    ActionStatus,
    build_assistant_message_content,
    dump_model_messages,
)
from assistant_runtime.app.assistant.usage import (
    merge_usage,
    response_service_tiers,
    usage_dict,
    with_auxiliary,
)

if TYPE_CHECKING:
    from assistant_runtime.app.assistant import SessionStore
    from assistant_runtime.app.streaming._turn import TurnPlan
    from assistant_runtime.services.history.interface import HistoryProcessor


def _keep_row_auxiliary(
    usage: dict[str, Any] | None, row: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Keep row-owned memory totals and auxiliary sections missing from a snapshot."""
    existing = ((row or {}).get("usage") or {}).get("auxiliary") or {}
    for name, part in existing.items():
        auxiliary = (usage or {}).get("auxiliary") or {}
        if name == "working_memory" and name in auxiliary:
            # Background extraction owns this cumulative total; the turn's
            # earlier copy must neither overwrite it nor be added a second time.
            usage = {**usage, "auxiliary": {**auxiliary, name: part}}
        elif name not in auxiliary:
            usage = with_auxiliary(usage, name, part)
    return usage


@dataclass
class RunSnapshot:
    """What one agent run left behind, carried across steering follow-ups."""

    all_messages: list[ModelMessage] = field(default_factory=list)
    assistant_messages: list[ModelMessage] = field(default_factory=list)
    assistant_segments: list[dict[str, Any]] | None = None
    run_usage: Any = None
    usage: dict[str, int] | None = None
    service_tiers: list[dict[str, Any]] | None = None
    output: Any = None
    final_output: str | None = None
    pending_tool_call: dict[str, Any] | None = None
    persisted: bool = False
    cancelled: bool = False
    # How calls left without a result by this turn are recorded (see ActionStatus).
    interrupted_status: ActionStatus = "superseded"
    history: HistoryProcessor | None = None
    # The usage written on the assistant row: the run's own plus auxiliary work.
    stored_usage: dict[str, Any] | None = None
    # What the latest run of the turn was given (app/assistant/prompt_record).
    prompt_record: dict[str, Any] | None = None
    prompt_snapshot: str | None = None


class _IncompleteRun:
    """Captured native messages and usage when a run raises without a result."""

    output = None

    def __init__(self, captured: list[ModelMessage], usage: Any, run_id: str) -> None:
        self._messages = list(captured)
        self._run_id = run_id
        self.usage = usage

    def all_messages(self) -> list[ModelMessage]:
        return list(self._messages)

    def new_messages(self) -> list[ModelMessage]:
        return [m for m in self._messages if getattr(m, "run_id", None) == self._run_id]


def _without_user_prompt(messages: list[ModelMessage]) -> list[ModelMessage]:
    """A message turn's messages after its user prompt, which the user row keeps.

    Pydantic AI sends the prompt in the run's first request; steering it
    delivers follows in requests of its own.
    """
    if not messages or not isinstance(messages[0], ModelRequest):
        return messages
    first = messages[0]
    rest = [part for part in first.parts if not isinstance(part, UserPromptPart)]
    return [replace(first, parts=rest), *messages[1:]] if rest else messages[1:]


def _without_steering(messages: list[ModelMessage], pending_ids: set[str]) -> list[ModelMessage]:
    """``messages`` without the steering requests (``build_steering_request``) still pending."""
    if not pending_ids:
        return messages
    return [
        message
        for message in messages
        if not (
            isinstance(message, ModelRequest)
            and pending_ids.intersection((message.metadata or {}).get("steering_ids") or ())
        )
    ]


def capture_result(plan: TurnPlan, state: RunSnapshot, result: Any) -> None:
    """Persist native history, including snapshots attached during teardown."""
    state.all_messages = list(result.all_messages())
    # new_messages(), not index slicing: pydantic-ai may merge consecutive
    # ModelRequests while cleaning the history, shrinking the list.
    new_messages = result.new_messages()
    if plan.kind == "message" and not state.assistant_messages:
        new_messages = _without_user_prompt(new_messages)
    if plan.accepted_tool_result is not None and any(
        isinstance(part, ToolReturnPart) and part.tool_call_id == plan.request.tool_call_id
        for message in new_messages
        for part in message.parts
    ):
        # The result was saved before setup; native resumption now includes it.
        state.assistant_messages = [
            message
            for message in state.assistant_messages
            if message is not plan.accepted_tool_result
        ]
    state.assistant_messages.extend(new_messages)
    state.run_usage = result.usage
    state.usage = merge_usage(plan.prior_usage, usage_dict(result))
    # Counts are cumulative within a turn; tiers arrive per response. Restored host
    # history may lack provider_details, so retain the prior usage prefix explicitly.
    if state.service_tiers is None:
        state.service_tiers = list((plan.prior_usage or {}).get("service_tiers") or [])
    state.service_tiers.extend(response_service_tiers(new_messages))
    if state.usage is not None and state.service_tiers:
        state.usage["service_tiers"] = list(state.service_tiers)
    if not isinstance(result, RunCancelled):
        state.output = result.output


async def persist_snapshot(sessions: SessionStore, plan: TurnPlan, state: RunSnapshot) -> None:
    """Create or extend the assistant row with everything run so far."""
    if plan.request.output_mode == "host_tools":
        state.assistant_messages = [
            replace(m, parts=[p for p in m.parts if not isinstance(p, (TextPart, ThinkingPart))])
            if isinstance(m, ModelResponse)
            else m
            for m in state.assistant_messages
        ]
    # Steering whose model request did not complete stays queued and is sent
    # again; the snapshot must not keep a copy of it as well.
    state.assistant_messages = _without_steering(
        state.assistant_messages, set(plan.session_context.get("pending_steering_ids") or ())
    )
    content, segments, timestamp = build_assistant_message_content(
        state.assistant_messages, interrupted_status=state.interrupted_status
    )
    state.assistant_segments = segments
    # The history policy's summarisation calls are this turn's auxiliary
    # model work; state.usage stays the run's own so this is idempotent.
    summarisation = (
        usage_dict(state.history.usage)
        if state.history is not None and state.history.usage.has_values()
        else None
    )
    state.stored_usage = with_auxiliary(state.usage, "summarization", summarisation)
    # Auxiliary sections already on the row (a background extraction that
    # finished after this turn took its snapshot) are kept, not erased.
    if state.prompt_record is not None and state.prompt_snapshot is not None:
        await sessions.save_prompt_snapshot(
            plan.session_id, state.prompt_record["snapshot_hash"], state.prompt_snapshot
        )
    model_messages = dump_model_messages(state.assistant_messages)
    if plan.assistant_parent_id is not None and not state.persisted:
        await sessions.register_assistant_message(
            plan.session_id,
            message_id=plan.assistant_message_id,
            parent_id=plan.assistant_parent_id,
            content=content,
            segments=segments,
            usage=state.stored_usage,
            created_at=timestamp,
            prompt=state.prompt_record,
            model_messages=model_messages,
        )
    else:
        record = await sessions.update_message(
            plan.session_id,
            plan.assistant_message_id,
            content=content,
            segments=segments,
            usage=lambda current: _keep_row_auxiliary(state.stored_usage, {"usage": current}),
            prompt=state.prompt_record,
            model_messages=model_messages,
        )
        state.stored_usage = record.get("usage")
    state.persisted = True


def mark_unanswered_calls(state: RunSnapshot) -> bool:
    """Append interrupted returns for calls without a result; True when any were added."""
    returned = {
        part.tool_call_id
        for message in state.assistant_messages
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    interrupted = [
        ToolReturnPart(
            tool_name=part.tool_name,
            tool_call_id=part.tool_call_id,
            content="[Tool execution was interrupted; its external outcome is unknown.]",
            outcome="interrupted",
        )
        for message in state.assistant_messages
        for part in message.parts
        if isinstance(part, ToolCallPart) and part.tool_call_id not in returned
    ]
    if not interrupted:
        return False
    state.assistant_messages.append(ModelRequest(parts=interrupted))
    return True
