"""``assistant-runtime chat``: the runtime in-process, streamed to the terminal.

The command builds the same FastAPI application the server runs, opens its
lifespan (so every service, tool and prompt artifact is wired exactly as in
production), and feeds messages straight into the streaming service. Events
are rendered as they arrive: text is printed as it streams, tool calls are
shown on their own lines, and thinking is shown only with ``--show-thinking``.

Tools the runtime defers to a host application (a dashboard's ``navigate``,
for example) have no host here; the chat answers them with an error result so
the model can carry on.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import IO, Any

from assistant_runtime.app.assistant.models import AssistantRequest, RequestConfigOverride

EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})
NO_HOST_RESULT = {
    "success": False,
    "error": "No host application is attached to this terminal chat; this tool is unavailable.",
    "error_code": "NO_HOST_ATTACHED",
}
MAX_HOST_TOOL_ROUNDS = 3


def _short(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class TurnRenderer:
    """Turn stream events into terminal output. Stateless across turns except for the cursor."""

    def __init__(self, out: IO[str] | None = None, *, show_thinking: bool = False) -> None:
        self._out = out or sys.stdout
        self._show_thinking = show_thinking
        self._line_open = False
        self._thinking_open = False
        self.pending_tool_call: dict[str, Any] | None = None
        self.error: str | None = None
        self.final_content: str | None = None

    def start_turn(self) -> None:
        self._line_open = False
        self._thinking_open = False
        self.pending_tool_call = None
        self.error = None
        self.final_content = None

    def _write(self, text: str) -> None:
        self._out.write(text)
        self._out.flush()

    def _newline_if_open(self) -> None:
        if self._line_open or self._thinking_open:
            self._write("\n")
            self._line_open = False
            self._thinking_open = False

    def _line(self, text: str) -> None:
        self._newline_if_open()
        self._write(text + "\n")

    def handle(self, event: dict[str, Any]) -> None:
        """Render one streaming event."""
        kind = event.get("type", "")
        if kind == "text_delta":
            if self._thinking_open:
                self._write("\n")
                self._thinking_open = False
            self._write(str(event.get("content", "")))
            self._line_open = True
        elif kind == "thinking_delta":
            if self._show_thinking:
                self._write(str(event.get("content", "")))
                self._thinking_open = True
        elif kind == "tool_call":
            self._line(f"[tool] {event.get('tool_name')}({_short(event.get('arguments', {}))})")
        elif kind == "tool_result":
            suffix = f" in {event['duration_ms']}ms" if "duration_ms" in event else ""
            self._line(f"[tool] {event.get('tool_name')} done{suffix}")
        elif kind == "tool_error":
            self._line(f"[tool] {event.get('tool_name')} failed: {_short(event.get('error'))}")
        elif kind == "error":
            self.error = str(event.get("message", "unknown error"))
            self._line(f"[error] {self.error}")
        elif kind == "final_response":
            self.pending_tool_call = event.get("pending_tool_call")
            self.final_content = event.get("content")
            if event.get("error"):
                self.error = self.error or str(event.get("content") or "request failed")
            if not event.get("streamed") and event.get("content"):
                self._line(str(event["content"]))
            elif self._line_open:
                self._newline_if_open()
        # agent_status, tool_status and debug_* events carry nothing to show.


async def run_turn(
    streaming: Any,
    request: AssistantRequest,
    renderer: TurnRenderer,
) -> str | None:
    """Stream one request (and any host-tool continuations) to the renderer.

    Returns the final assistant text, or None when the turn ended in an error.
    """
    rounds = 0
    while True:
        renderer.start_turn()
        async for event in streaming.stream_message(request):
            renderer.handle(event)
            if event.get("type") == "agent_status" and event.get("status") == "completed":
                break
        pending = renderer.pending_tool_call
        if renderer.error is not None:
            return None
        if not pending:
            return renderer.final_content
        rounds += 1
        if rounds > MAX_HOST_TOOL_ROUNDS:
            renderer.handle(
                {"type": "error", "message": "model kept requesting host tools; giving up"}
            )
            return None
        renderer.handle(
            {
                "type": "tool_error",
                "tool_name": pending.get("tool_name"),
                "error": NO_HOST_RESULT["error"],
            }
        )
        request = AssistantRequest(
            id=str(uuid.uuid4()),
            session_id=request.session_id,
            content="",
            tool_call_id=str(pending.get("call_id", "")),
            tool_result=dict(NO_HOST_RESULT),
        )


def build_request(session_id: str, content: str, model: str | None) -> AssistantRequest:
    """A standard request for one user message."""
    config = RequestConfigOverride(default_model=model) if model else None
    return AssistantRequest(
        id=str(uuid.uuid4()), session_id=session_id, content=content, config=config
    )


async def _read_line(prompt: str) -> str | None:
    try:
        return await asyncio.to_thread(input, prompt)
    except (EOFError, KeyboardInterrupt):
        return None


async def chat_loop(args: argparse.Namespace) -> int:
    """Open the app lifespan and run the read/stream loop."""
    from assistant_runtime.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        streaming = app.state.streaming_service
        session_id = args.session or f"cli-{uuid.uuid4().hex[:12]}"
        await streaming.warm_session(session_id)
        renderer = TurnRenderer(show_thinking=bool(args.show_thinking))

        if args.message:
            result = await run_turn(
                streaming, build_request(session_id, args.message, args.model), renderer
            )
            return 0 if result is not None else 1

        db = getattr(app.state, "database_service", None)
        persistence = "database" if db is not None and getattr(db, "healthy", False) else "memory"
        model = args.model or app.state.runtime_settings.get("default_model")
        print(f"assistant-runtime chat · model {model} · session {session_id} · {persistence}")
        print("Type a message and press Enter. /exit to quit.")
        while True:
            line = await _read_line("\nyou> ")
            if line is None or line.strip().lower() in EXIT_COMMANDS:
                print()
                return 0
            if not line.strip():
                continue
            print()
            await run_turn(streaming, build_request(session_id, line, args.model), renderer)


def cmd_chat(args: argparse.Namespace) -> int:
    """Entry point for ``assistant-runtime chat``."""
    os.environ.setdefault("LOG_LEVEL", "DEBUG" if args.verbose else "WARNING")
    try:
        return asyncio.run(chat_loop(args))
    except KeyboardInterrupt:
        return 130
