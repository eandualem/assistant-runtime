"""Walk one pydantic-ai agent run and turn its graph nodes into stream events.

``ModelRequestNode`` streams text and thinking deltas; ``CallToolsNode``
emits ``tool_call`` (with complete arguments), executes the tools and emits
``tool_result`` / ``tool_error``. Steering queued while tools ran is appended
to the next model request so the model sees it in the same run.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
)
from pydantic_graph import End

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


async def iterate_run(
    run: Any,
    coordinator: EventCoordinator,
    *,
    config: StreamingConfig,
    tools: ToolService,
    sessions: SessionStore,
    session_id: str,
    session_context: dict[str, Any],
    emit_debug: bool,
    suppress_tool_call_ids: set[str] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield stream events for every node of ``run`` until it ends."""
    suppressed = suppress_tool_call_ids or set()
    pending_image_sanitize = False
    while True:
        node = run.next_node
        if isinstance(node, End):
            return
        if isinstance(node, ModelRequestNode):
            async for event in _stream_model_request(node, run, coordinator, config):
                yield event
            if emit_debug:
                coordinator.flush_thinking()
            coordinator.clear_content_segment()
            if pending_image_sanitize:
                # The model has now seen the look_at_screen image; strip it so
                # later requests in this run do not resend the binary content.
                sanitize_image_tool_returns(run.ctx.state.message_history)
                pending_image_sanitize = False
            await run.next(node)
        elif isinstance(node, CallToolsNode):
            calls = [
                tc for tc in node.model_response.tool_calls if tc.tool_call_id not in suppressed
            ]
            for tc in calls:
                try:
                    args = tc.args_as_dict()
                except Exception:
                    args = {}
                category = "frontend" if tools.is_host_tool(tc.tool_name) else "backend"
                yield _track(
                    coordinator,
                    make_tool_call_event(tc.tool_name, args, tc.tool_call_id, category=category),
                )

            tool_start = time.monotonic()
            next_node = await run.next(node)
            tool_duration_ms = (time.monotonic() - tool_start) * 1000

            if isinstance(next_node, ModelRequestNode) and calls:
                for tc in calls:
                    raw = _tool_return_content(next_node, tc.tool_call_id)
                    is_error, error_msg = _tool_error(raw)
                    if is_error:
                        yield _track(
                            coordinator,
                            make_tool_error_event(tc.tool_name, error_msg, tc.tool_call_id),
                        )
                    result_str = error_msg if is_error else (str(raw) if raw is not None else "")
                    yield _track(
                        coordinator,
                        make_tool_result_event(
                            tc.tool_name,
                            result_str,
                            tc.tool_call_id,
                            duration_ms=tool_duration_ms,
                            invalidates=tools.get_tool_invalidates(tc.tool_name),
                        ),
                    )
                for part in next_node.request.parts:
                    if isinstance(part, RetryPromptPart):
                        yield _track(
                            coordinator,
                            make_tool_error_event(
                                getattr(part, "tool_name", None) or "unknown",
                                str(part.content) if part.content else "Validation failed",
                                getattr(part, "tool_call_id", None) or "",
                            ),
                        )
                await deliver_steering_into_request(
                    sessions, session_id, session_context, next_node=next_node
                )

            if any(tc.tool_name == "look_at_screen" for tc in calls):
                pending_image_sanitize = True
        else:
            await run.next(node)


async def deliver_steering_into_request(
    sessions: SessionStore,
    session_id: str,
    session_context: dict[str, Any],
    *,
    next_node: ModelRequestNode,
) -> list[SteeringRecord]:
    """Append queued steering to the next model request and mark it delivered."""
    if not session_context.get("pending_steering_ids"):
        return []
    delivered = await sessions.deliver_pending_steering(session_id)
    if not delivered:
        return []
    next_node.request.parts.extend(build_steering_request(delivered).parts)
    return delivered


def _track(coordinator: EventCoordinator, event: dict[str, Any]) -> dict[str, Any]:
    """Tool events are both streamed and kept in the debug trace."""
    coordinator.track_debug(event)
    return coordinator.track(event)


async def _stream_model_request(
    node: ModelRequestNode, run: Any, coordinator: EventCoordinator, config: StreamingConfig
) -> AsyncIterator[dict[str, Any]]:
    """Stream text and thinking deltas from one model request.

    Tool calls are not emitted here: at ``PartStartEvent`` time their
    arguments are still empty. Large ``PartStartEvent`` payloads (common with
    Gemini thinking) are chunked so the browser does not stall on one event.
    """
    async with node.stream(run.ctx) as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                if isinstance(event.part, ThinkingPart) and event.part.content:
                    async for e in _chunked(
                        event.part.content, coordinator.emit_thinking_delta, config
                    ):
                        yield e
                elif isinstance(event.part, TextPart) and event.part.content:
                    async for e in _chunked(
                        event.part.content, coordinator.emit_text_delta, config
                    ):
                        yield e
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, ThinkingPartDelta) and event.delta.content_delta:
                    yield coordinator.emit_thinking_delta(event.delta.content_delta)
                elif isinstance(event.delta, TextPartDelta) and event.delta.content_delta:
                    yield coordinator.emit_text_delta(event.delta.content_delta)


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


def _tool_return_content(next_node: ModelRequestNode, tool_call_id: str) -> Any:
    try:
        for part in next_node.request.parts:
            if isinstance(part, ToolReturnPart) and part.tool_call_id == tool_call_id:
                return part.content
    except Exception as exc:
        logger.debug("Tool result extraction failed", tool_call_id=tool_call_id, error=str(exc))
    return None


def _tool_error(content: Any) -> tuple[bool, str]:
    """Handlers report failure as a dict with ``error`` or ``error_code``."""
    if isinstance(content, dict) and ("error" in content or "error_code" in content):
        return True, str(content.get("error", content.get("error_code", "Unknown error")))
    return False, ""
