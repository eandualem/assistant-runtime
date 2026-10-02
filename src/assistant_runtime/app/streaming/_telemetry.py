"""Streaming diagnostics, failure envelopes and best-effort trace persistence."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

from assistant_runtime.app.assistant.usage import (
    cache_counts,
    usage_dict,
)
from assistant_runtime.app.streaming._coordinator import EventCoordinator
from assistant_runtime.app.streaming._event_builder import (
    make_debug_agent_config_event,
    make_debug_error_event,
    make_debug_history_event,
    make_debug_system_prompt_event,
    make_debug_tool_selection_event,
    make_debug_usage_event,
    make_error_event,
)
from assistant_runtime.app.streaming.exceptions import InvalidDecisionError
from assistant_runtime.services.llm.exceptions import LLMCallError, classify_llm_error

if TYPE_CHECKING:
    from assistant_runtime.app.assistant.models import AgentSetupContext
    from assistant_runtime.app.streaming._snapshot import RunSnapshot
    from assistant_runtime.services.database.interface import DatabaseService

_LLM_ERROR_TYPES = {
    "RATE_LIMIT": "rate_limit",
    "SERVER_ERROR": "provider_error",
    "CONNECTION_ERROR": "connection_error",
    "TIMEOUT": "timeout",
    "AUTH_ERROR": "provider_auth",
    "CLIENT_ERROR": "provider_client_error",
}


def format_error_message(message: str, trace_id: str | None) -> str:
    """Append the trace id so client-visible errors can be correlated with traces."""
    if trace_id is None or "[trace_id:" in message:
        return message
    return f"{message} [trace_id: {trace_id}]"


def _describe_error(exc: Exception, *, detail: bool = True) -> tuple[str, str, bool]:
    """``(message, error_type, retry_allowed)`` for an exception raised by the run.

    ``detail`` (``STREAMING__CLIENT_ERROR_DETAIL``) decides whether the
    exception text itself reaches the client; the log always has it.
    """
    if isinstance(exc, InvalidDecisionError):
        return str(exc), "invalid_decision", False
    if isinstance(exc, LLMCallError):
        llm_error: LLMCallError | None = exc
    else:
        classified = classify_llm_error(exc)
        llm_error = None if classified.error_category == "UNKNOWN" else classified
    if llm_error is None:
        message = f"Request failed: {exc.__class__.__name__}: {exc}" if detail else "Request failed"
        return message, "internal", False
    error_type = _LLM_ERROR_TYPES.get(llm_error.error_category, "provider_error")
    message = str(llm_error) if detail else f"LLM call failed ({llm_error.error_category})"
    return message, error_type, llm_error.retry_allowed


def _persistence_failure(what: str, exc: BaseException, *, detail: bool) -> str:
    """The client-visible text for a snapshot that could not be saved."""
    return f"{what} could not be saved: {exc}" if detail else f"{what} could not be saved"


def _setup_debug_events(ctx: AgentSetupContext, session_id: str) -> list[dict[str, Any]]:
    tools = ctx.available_tools
    return [
        make_debug_tool_selection_event(
            page=tools.page,
            backend_count=len(tools.backend_tools),
            filtered_out=tools.filtered_out_count,
            tool_names=tools.tool_names,
            tools=[
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters_schema": t.parameters_schema,
                }
                for t in tools.backend_tools
            ],
        ),
        make_debug_system_prompt_event(
            total_length=len(ctx.prompt_result.content),
            fragment_count=len(ctx.prompt_result.fragments),
            fragments=ctx.prompt_result.fragments,
            content=ctx.prompt_result.content,
        ),
        make_debug_agent_config_event(
            model=ctx.resolved_model,
            output_type="str",
            thinking_budget=ctx.effective_config.thinking_budget,
            temperature=ctx.effective_config.temperature,
            session_id=session_id,
        ),
    ]


def _history_debug_event(state: RunSnapshot) -> dict[str, Any] | None:
    """Describe the model input the history policy last produced, if it ran."""
    result = state.history.result if state.history is not None else None
    if result is None:
        return None
    return make_debug_history_event(
        message_count=result.message_count,
        estimated_tokens=result.estimated_tokens,
        was_compacted=result.was_compacted,
        compacted_from=result.compacted_from,
        messages=result.message_summaries,
    )


def _usage_debug_event(run_usage: Any) -> dict[str, Any]:
    snapshot = usage_dict(run_usage) or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    cache_read, cache_write = cache_counts(run_usage)
    return make_debug_usage_event(
        input_tokens=snapshot["input_tokens"],
        output_tokens=snapshot["output_tokens"],
        cache_read=cache_read,
        cache_write=cache_write,
        requests=getattr(run_usage, "requests", 0) or 0,
        total=snapshot["total_tokens"],
    )


def _outcome_events(
    coordinator: EventCoordinator,
    *,
    session_id: str,
    message: str,
    error_type: str,
    retry_allowed: bool,
    trace_id: str,
    model: str,
    phase: str,
    emit_debug: bool,
    message_id: str | None = None,
    content: str | None = None,
    usage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """The one terminal lifecycle of a turn that did not finish normally.

    ``final_response(error=True)``, the debug error (always traced, sent
    only with ``emit_debug``) and the terminal ``error``. A saved snapshot
    (cancellation, usage limit) names its ``message_id`` and ``usage``;
    ``content`` stays empty, only native middleware decides what text a
    client sees. Like the other terminal events, the error is not counted
    against the event limit: the client must always receive it.
    """
    final = coordinator.try_final_response(
        content,
        model,
        session_id=session_id,
        message_id=message_id,
        trace_id=trace_id,
        error=True,
        error_type=error_type,
        usage=usage,
    )
    debug = coordinator.track_debug(
        make_debug_error_event(
            message,
            error_type=error_type,
            retry_allowed=retry_allowed,
            trace_id=trace_id,
            model=model,
            phase=phase,
        )
    )
    events: list[dict[str, Any]] = []
    if final:
        events.append(final)
    if emit_debug:
        events.append(debug)
    error = make_error_event(
        format_error_message(message, trace_id),
        error_type=error_type,
        trace_id=trace_id,
        terminal=True,
        retry_allowed=retry_allowed,
    )
    events.append(error)
    return events


async def save_trace(
    database: DatabaseService | None,
    session_id: str,
    trace_events: list[dict[str, Any]],
    *,
    trace_id: str | None = None,
    user_message: str | None = None,
    duration_ms: float | None = None,
) -> None:
    """Persist collected debug events as a trace row (never the screenshot). Best-effort."""
    if database is None or not database.healthy or not trace_events:
        return
    try:
        from assistant_runtime.services.database.repositories import TraceRepository

        async with database.session_context() as db_session:
            await TraceRepository(db_session).create(
                trace_id=trace_id or str(uuid.uuid4()),
                session_id=session_id,
                events=trace_events,
                user_message=user_message,
                duration_ms=duration_ms,
            )
    except Exception as e:
        logger.warning("Failed to persist trace", session_id=session_id, error=str(e))
