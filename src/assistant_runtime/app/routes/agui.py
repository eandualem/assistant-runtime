"""AG-UI endpoint — ``POST /agui``, the runtime's turn pipeline behind the AG-UI protocol."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from assistant_runtime.app.access.deps import PrincipalDep
from assistant_runtime.app.access.exceptions import AccessDeniedError
from assistant_runtime.app.assistant.deps import AssistantServiceDep
from assistant_runtime.app.streaming.deps import StreamingServiceDep
from assistant_runtime.principal import can_access_session

router = APIRouter()


def _load_bridge():
    """Import the protocol mapping lazily; the ``ag-ui`` extra is optional."""
    from assistant_runtime.app.routes import _agui

    return _agui


@router.post("/agui")
async def agui_run(
    request: Request,
    streaming: StreamingServiceDep,
    assistant: AssistantServiceDep,
    principal: PrincipalDep,
):
    """Run one turn for an AG-UI ``RunAgentInput`` and stream AG-UI events.

    ``threadId`` is the session. A trailing tool message continues the
    session's pending host action; a trailing user message is a new message.
    Frontend ``tools`` become host actions for the turn. Errors that end the
    turn are ``RUN_ERROR`` events; a client disconnect cancels the turn.
    """
    try:
        bridge = _load_bridge()
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="AG-UI support needs the 'ag-ui' extra: pip install 'assistant-runtime[ag-ui]'",
        ) from exc

    try:
        run_input = bridge.parse_run_input(await request.body())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc

    sessions = assistant.get_session_store()
    context = await sessions.get_context_if_exists_async(run_input.thread_id)
    if context is not None and not can_access_session(principal, context.get("owner_id")):
        raise AccessDeniedError(f"Session '{run_input.thread_id}' belongs to another principal")

    try:
        assistant_request = bridge.build_assistant_request(run_input, context)
    except (bridge.AGUIRequestError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return bridge.agui_response(
        streaming,
        run_input,
        assistant_request,
        principal=principal,
        accept=request.headers.get("accept"),
    )
