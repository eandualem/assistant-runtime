"""Map Pydantic AI's public event stream to the application's wire events."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    EnqueuedMessagesEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)

from assistant_runtime.app.assistant._serialization import (
    SteeringRecord,
    build_steering_request,
    sanitize_image_tool_returns,
)
from assistant_runtime.app.streaming._coordinator import EventCoordinator
from assistant_runtime.app.streaming._event_builder import (
    make_tool_call_event,
    make_tool_error_event,
    make_tool_result_event,
)

if TYPE_CHECKING:
    from assistant_runtime.app.assistant._session_store import SessionStore
    from assistant_runtime.app.streaming.config import StreamingConfig
    from assistant_runtime.services.tools.interface import ToolService


class TurnPolicy(AbstractCapability):
    """Apply session steering and screen retention at supported node boundaries.

    Pydantic AI owns node execution and capability ordering. The application
    only supplies its queue and retention policy through public hooks.
    """

    def __init__(self, sessions: SessionStore, session_id: str, session_context: dict[str, Any]):
        self.sessions = sessions
        self.session_id = session_id
        self.session_context = session_context
        self.pending_image_sanitize = False
        self._enqueued_steering: dict[str, list[str]] = {}
        self._inflight_steering: set[str] = set()
        self._consumed_steering: set[str] = set()

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        if Agent.is_user_prompt_node(node):
            # Includes steering retained after an interrupted previous turn.
            await self._enqueue_steering(ctx)
        return node

    async def on_event(self, ctx: RunContext[Any], *, event: Any) -> None:
        if isinstance(event, EnqueuedMessagesEvent):
            self._consumed_steering.update(self._enqueued_steering.pop(event.enqueue_id, []))

    async def _enqueue_steering(self, ctx: RunContext[Any]) -> None:
        enqueue_id, records = await enqueue_pending_steering(
            self.sessions,
            self.session_id,
            self.session_context,
            run=ctx,
            exclude_ids=self._inflight_steering,
        )
        if enqueue_id is not None:
            ids = [record["id"] for record in records]
            self._enqueued_steering[enqueue_id] = ids
            self._inflight_steering.update(ids)

    async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
        if Agent.is_model_request_node(node):
            if self.pending_image_sanitize:
                # The first request after look_at_screen has consumed the image.
                sanitize_image_tool_returns(ctx.messages)
                self.pending_image_sanitize = False
            if self._consumed_steering and Agent.is_call_tools_node(result):
                # Native enqueue insertion alone does not prove a model request
                # completed. Keep records pending if that request is interrupted:
                # application message segments do not persist steering prompts.
                await self.sessions.mark_steering_delivered(
                    self.session_id, list(self._consumed_steering)
                )
                self._consumed_steering.clear()
        elif Agent.is_call_tools_node(node):
            if Agent.is_model_request_node(result):
                await self._enqueue_steering(ctx)
            if any(tc.tool_name == "look_at_screen" for tc in node.model_response.tool_calls):
                self.pending_image_sanitize = True
        return result


async def iterate_run(
    stream: Any,
    coordinator: EventCoordinator,
    *,
    config: StreamingConfig,
    tools: ToolService,
    emit_debug: bool,
    suppress_tool_call_ids: set[str] | None = None,
    host_tool_names: set[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Translate native events without executing or inspecting graph nodes.

    ``host_tool_names`` are the host tools of this turn (configured ones
    plus those the request declared); calls to them are ``category: host``.
    """
    suppressed = suppress_tool_call_ids or set()
    host_names = host_tool_names if host_tool_names is not None else set()
    tool_started: dict[str, float] = {}
    async for event in stream:
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, ThinkingPart) and event.part.content:
                async for chunk in _chunked(
                    event.part.content, coordinator.emit_thinking_delta, config
                ):
                    yield chunk
            elif isinstance(event.part, TextPart) and event.part.content:
                async for chunk in _chunked(
                    event.part.content, coordinator.emit_text_delta, config
                ):
                    yield chunk
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
                yield coordinator.emit_thinking_delta(event.delta.content_delta)
            elif isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                yield coordinator.emit_text_delta(event.delta.content_delta)
        elif isinstance(event, PartEndEvent) and event.next_part_kind is None:
            # A retry may start another response without any intervening tool event.
            if emit_debug:
                coordinator.flush_thinking()
            coordinator.clear_content_segment()
        elif isinstance(event, FunctionToolCallEvent):
            tc = event.part
            if tc.tool_call_id in suppressed:
                continue
            if emit_debug:
                coordinator.flush_thinking()
            tool_started[tc.tool_call_id] = time.monotonic()
            is_host = tc.tool_name in host_names or tools.is_host_tool(tc.tool_name)
            category = "host" if is_host else "backend"
            yield _track(
                coordinator,
                make_tool_call_event(
                    tc.tool_name, tc.args_as_dict(), tc.tool_call_id, category=category
                ),
            )
        elif isinstance(event, FunctionToolResultEvent):
            part = event.part
            if part.tool_call_id in suppressed:
                continue
            started = tool_started.pop(part.tool_call_id, None)
            duration_ms = (time.monotonic() - started) * 1000 if started is not None else 0.0
            is_error, error_msg = _tool_error(part.content)
            if isinstance(part, RetryPromptPart):
                is_error, error_msg = True, part.model_response()
            elif part.outcome == "failed":
                is_error, error_msg = True, part.model_response_str(wrap_if_error=False)
            tool_name = part.tool_name or "unknown"
            if is_error:
                yield _track(
                    coordinator, make_tool_error_event(tool_name, error_msg, part.tool_call_id)
                )
            yield _track(
                coordinator,
                make_tool_result_event(
                    tool_name,
                    error_msg
                    if is_error
                    else (str(part.content) if part.content is not None else ""),
                    part.tool_call_id,
                    duration_ms=duration_ms,
                    invalidates=tools.get_tool_invalidates(tool_name),
                ),
            )
    if emit_debug:
        coordinator.flush_thinking()
    coordinator.clear_content_segment()


async def enqueue_pending_steering(
    sessions: SessionStore,
    session_id: str,
    session_context: dict[str, Any],
    *,
    run: Any,
    exclude_ids: set[str] | None = None,
) -> tuple[str | None, list[SteeringRecord]]:
    """Enqueue pending records; acknowledge only after their model request succeeds."""
    if not session_context.get("pending_steering_ids"):
        return None, []
    pending = await sessions.list_pending_steering(session_id)
    records = [record for record in pending if record["id"] not in (exclude_ids or set())]
    if not records:
        return None, []
    enqueue_id = run.enqueue(build_steering_request(records), priority="asap")
    return enqueue_id, records


def _track(coordinator: EventCoordinator, event: dict[str, Any]) -> dict[str, Any]:
    """Tool events are both streamed and kept in the debug trace."""
    coordinator.track_debug(event)
    return coordinator.track(event)


async def _chunked(
    content: str, emit: Callable[[str], dict[str, Any]], config: StreamingConfig
) -> AsyncIterator[dict[str, Any]]:
    if len(content) <= config.part_start_chunk_threshold:
        yield emit(content)
        return
    size = config.part_start_chunk_size
    for i in range(0, len(content), size):
        yield emit(content[i : i + size])
        await asyncio.sleep(0)


def _tool_error(content: Any) -> tuple[bool, str]:
    """Handlers report failure as a dict with ``error`` or ``error_code``."""
    if isinstance(content, dict) and ("error" in content or "error_code" in content):
        return True, str(content.get("error", content.get("error_code", "Unknown error")))
    return False, ""
