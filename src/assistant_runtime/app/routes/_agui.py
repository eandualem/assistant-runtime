"""AG-UI on the runtime's own turn pipeline.

An AG-UI client posts a ``RunAgentInput`` (the thread, its transcript, the
frontend tools and state) and reads Server-Sent Events. This module maps that
input onto the runtime's session model and lets upstream's ``AGUIEventStream``
encode the turn's native Pydantic AI events:

- ``threadId`` is the session id; the server-side message tree stays
  authoritative and the resent transcript is not replayed into the model;
- the last message decides the turn: a ``ToolMessage`` is the continuation of
  the session's pending host action, a ``UserMessage`` is a new message
  appended to the active leaf;
- ``tools`` become request-declared host actions (``host_context.actions``),
  ``context`` becomes ``host_context.background``, a ``state`` object becomes
  the host context when it carries the contract ``version`` and lands in
  ``host_context.extensions.state`` otherwise; ``forwardedProps.config`` is
  the per-request tunable override.

Requires the ``ag-ui`` extra (``ag-ui-protocol``); the route imports this
module lazily and answers 501 without it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from ag_ui.core import (
    BinaryInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentUrlSource,
    RunAgentInput,
    TextInputContent,
    ToolMessage,
    UserMessage,
)
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.host_context import HOST_CONTEXT_VERSION

if TYPE_CHECKING:
    from starlette.responses import StreamingResponse

    from assistant_runtime.app.streaming.interface import StreamingService
    from assistant_runtime.principal import Principal


class AGUIRequestError(ValueError):
    """The run input cannot be mapped onto a turn (the client's mistake, 422)."""


class TurnFailedError(Exception):
    """The turn ended with a terminal runtime error; AG-UI reports it as ``RUN_ERROR``."""

    def __init__(self, message: str, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


def parse_run_input(body: bytes) -> RunAgentInput:
    """Validate the request body as AG-UI ``RunAgentInput`` (upstream parsing)."""
    return AGUIAdapter.build_run_input(body)


def build_assistant_request(
    run_input: RunAgentInput, session_context: dict[str, Any] | None
) -> AssistantRequest:
    """The runtime request for this run: a continuation or a new message."""
    if not run_input.messages:
        raise AGUIRequestError("AG-UI run input has no messages")
    last = run_input.messages[-1]
    base: dict[str, Any] = {
        "session_id": run_input.thread_id,
        "host_context": _host_context(run_input),
    }
    config = _forwarded_config(run_input)
    if config is not None:
        base["config"] = config

    if isinstance(last, ToolMessage):
        return AssistantRequest.model_validate(
            {
                **base,
                "id": last.id,
                "content": "",
                "tool_call_id": last.tool_call_id,
                "tool_result": _tool_result(last.content),
                "tool_outcome": "failed" if getattr(last, "error", None) else "success",
            }
        )
    if isinstance(last, UserMessage):
        text, attachments = _user_content(last)
        return AssistantRequest.model_validate(
            {
                **base,
                "id": last.id,
                "parent_id": (session_context or {}).get("active_leaf_id"),
                "content": text,
                "attachments": attachments,
            }
        )
    raise AGUIRequestError(
        f"AG-UI run input must end with a user or tool message, not '{last.role}'"
    )


def agui_response(
    service: StreamingService,
    run_input: RunAgentInput,
    request: AssistantRequest,
    *,
    principal: Principal,
    accept: str | None,
) -> StreamingResponse:
    """Stream the turn as AG-UI events, encoded by upstream for the ``Accept`` header."""
    event_stream = AGUIEventStream(run_input, accept=accept)
    return event_stream.streaming_response(
        event_stream.transform_stream(_native_events(service, request, principal))
    )


_DONE = object()


async def _native_events(
    service: StreamingService, request: AssistantRequest, principal: Principal
) -> AsyncIterator[Any]:
    """The turn's native events; a terminal runtime error ends the iterator by raising.

    The turn runs in its own task so the runtime finishes persistence even if
    the client goes away; closing this iterator (a disconnect) cancels the
    turn through the same path as any other consumer. The queue is unbounded
    like ``TurnControl.events``: a slow client must never stall the run or its
    snapshot write, and it holds at most one turn's events, which the run's
    usage limits and the stream timeout already cap.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def consume() -> None:
        try:
            async for event in service.stream_message(
                request, principal=principal, native_sink=queue.put_nowait
            ):
                if event.get("type") == "error" and event.get("terminal"):
                    queue.put_nowait(_terminal_error(event))
        except BaseException as exc:
            queue.put_nowait(exc)
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(consume())
    try:
        while (item := await queue.get()) is not _DONE:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _terminal_error(event: dict[str, Any]) -> BaseException:
    message = str(event.get("message") or "The turn failed")
    error_type = str(event.get("error_type") or "internal")
    if error_type == "cancelled":
        # AG-UI has no cancelled outcome; upstream closes the run as finished.
        return RunCancelled(message)
    return TurnFailedError(message, error_type)


# --- input mapping -----------------------------------------------------------


def _host_context(run_input: RunAgentInput) -> dict[str, Any]:
    state = run_input.state if isinstance(run_input.state, dict) else None
    context: dict[str, Any]
    if state and state.get("version") == HOST_CONTEXT_VERSION:
        context = dict(state)
    else:
        context = {"version": HOST_CONTEXT_VERSION}
        if state:
            context["extensions"] = {"state": state}
    if run_input.tools:
        actions = list(context.get("actions") or [])
        actions.extend(
            {
                "name": tool.name,
                "description": tool.description or f"Host action {tool.name}",
                "parameters": tool.parameters
                if isinstance(tool.parameters, dict)
                else {"type": "object", "properties": {}},
            }
            for tool in run_input.tools
        )
        context["actions"] = actions
    if run_input.context:
        background = dict(context.get("background") or {})
        background.update({item.description: item.value for item in run_input.context})
        context["background"] = background
    # Always a context, even an empty one: an AG-UI run declares its own tools
    # and state, so a previous run's host context must not carry over.
    return context


def _forwarded_config(run_input: RunAgentInput) -> dict[str, Any] | None:
    props = run_input.forwarded_props
    if isinstance(props, dict) and isinstance(props.get("config"), dict):
        return props["config"]
    return None


def _tool_result(content: str) -> Any:
    try:
        return json.loads(content)
    except (TypeError, ValueError):
        return content


def _user_content(message: UserMessage) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(message.content, str):
        return message.content, []
    texts: list[str] = []
    attachments: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextInputContent):
            texts.append(part.text)
        elif isinstance(part, ImageInputContent | DocumentInputContent):
            kind = "image" if isinstance(part, ImageInputContent) else "document"
            source = part.source
            attachment: dict[str, Any] = {"kind": kind, "purpose": "reference"}
            if isinstance(source, InputContentUrlSource):
                attachment["url"] = source.value
                attachment["media_type"] = source.mime_type
            else:
                attachment["media_type"] = source.mime_type
                attachment["data_uri"] = f"data:{source.mime_type};base64,{source.value}"
            attachments.append({k: v for k, v in attachment.items() if v is not None})
        elif isinstance(part, BinaryInputContent):
            # Deprecated in the protocol; still sent by older clients.
            attachment = {
                "kind": "image" if part.mime_type.startswith("image/") else "document",
                "purpose": "reference",
                "media_type": part.mime_type,
                "name": part.filename,
            }
            if part.url and part.url.startswith("data:"):
                attachment["data_uri"] = part.url
            elif part.url:
                attachment["url"] = part.url
            elif part.data:
                attachment["data_uri"] = f"data:{part.mime_type};base64,{part.data}"
            else:  # pragma: no cover - the protocol rejects a source-less part
                continue
            attachments.append({k: v for k, v in attachment.items() if v is not None})
        # Audio and video content have no attachment kind in the host contract
        # and are not mapped; documented as unsupported.
    return "\n".join(texts), attachments


__all__ = [
    "AGUIRequestError",
    "TurnFailedError",
    "agui_response",
    "build_assistant_request",
    "parse_run_input",
]
