"""PROTOTYPE: voice through the local Codex CLI's experimental realtime interface.

The runtime starts the official ``codex app-server`` and never reads or holds
credentials: the CLI authenticates with its own ChatGPT login. API-key
variables are removed from its environment so it cannot fall back to API
billing. Each call gets an ephemeral read-only thread; handoffs are
client-managed, so the Codex agent runs no task. The browser's WebRTC offer is
forwarded unchanged and the returned answer carries the audio directly between
the browser and the provider.

``thread/realtime/*`` is marked EXPERIMENTAL by the CLI and is not described in
the public app-server documentation; this transport is for evaluation only.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import tempfile
from typing import Any

from loguru import logger

from assistant_runtime.app.voice.exceptions import VoiceError

# The provider accepts v1 (with an alpha header the CLI does not send) or v3.
_REALTIME_VERSION = "v3"
# Guard for the prototype: never spend credits, stop before the included budget runs out.
USAGE_CEILING_PERCENT = 97
USAGE_CHECK_SECONDS = 15
# Voices the provider accepts for v3 (it rejects others, e.g. GPT-Live's default).
_CODEX_VOICES = frozenset(
    {"arbor", "breeze", "cove", "ember", "juniper", "maple", "sol", "spruce", "vale"}
)


class _AppServer:
    """Newline-delimited JSON-RPC over ``codex app-server --stdio``."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._threads: dict[str, asyncio.Queue] = {}
        self._reader: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def ensure_started(self) -> None:
        async with self._lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            env = {
                k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")
            }
            try:
                self._proc = await asyncio.create_subprocess_exec(
                    "codex",
                    "app-server",
                    "--stdio",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=env,
                    limit=2**24,
                )
            except FileNotFoundError as exc:
                raise VoiceError(
                    "Codex voice needs the Codex CLI on PATH", 503, allocation_status="rejected"
                ) from exc
            self._reader = asyncio.create_task(self._read())
            await self.request(
                "initialize",
                {
                    "clientInfo": {"name": "assistant-runtime", "version": "prototype"},
                    "capabilities": {"experimentalApi": True},
                },
            )
            await self._write({"jsonrpc": "2.0", "method": "initialized"})
            account = await self.request("account/read", {})
            if (account.get("account") or {}).get("type") != "chatgpt":
                raise VoiceError(
                    "Codex voice requires the Codex CLI to be signed in with ChatGPT",
                    503,
                    allocation_status="rejected",
                )

    async def _write(self, message: dict) -> None:
        assert self._proc is not None
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(message) + "\n").encode())
        await self._proc.stdin.drain()

    async def _read(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        while line := await self._proc.stdout.readline():
            with contextlib.suppress(ValueError):
                message = json.loads(line)
                ident = message.get("id")
                if ident in self._pending and ("result" in message or "error" in message):
                    self._pending.pop(ident).set_result(message)
                    continue
                thread = (message.get("params") or {}).get("threadId")
                if thread in self._threads:
                    self._threads[thread].put_nowait(message)
        for future in self._pending.values():
            if not future.done():
                future.set_exception(VoiceError("Codex app-server exited", 502))
        for queue in self._threads.values():
            queue.put_nowait(
                {"method": "thread/realtime/closed", "params": {"reason": "app_server_exit"}}
            )

    async def request(self, method: str, params: dict) -> Any:
        ident = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        self._pending[ident] = future
        await self._write({"jsonrpc": "2.0", "id": ident, "method": method, "params": params})
        message = await asyncio.wait_for(future, self.timeout)
        if "error" in message:
            raise VoiceError(f"Codex {method} failed", 502, allocation_status="rejected")
        return message.get("result") or {}

    async def usage_problem(self) -> str | None:
        """Why included usage must not be spent now, or None when it may."""
        limits = await self.request("account/rateLimits/read", {})
        snapshots = limits.get("rateLimitsByLimitId") or {"default": limits.get("rateLimits")}
        for snapshot in snapshots.values():
            if not snapshot:
                continue
            balance = snapshot.get("credits") or {}
            if (
                balance.get("hasCredits")
                or balance.get("unlimited")
                or balance.get("balance") not in (None, "0")
            ):
                return "the account has credits that could be charged"
            if snapshot.get("spendControlReached"):
                return "spend control is reached"
            for window in ("primary", "secondary"):
                if (snapshot.get(window) or {}).get("usedPercent", 0) >= USAGE_CEILING_PERCENT:
                    return f"{window} usage is at or above {USAGE_CEILING_PERCENT}%"
        if limits.get("ordinaryUsageAllowed") is False:
            return "included usage is not allowed"
        return None

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        return self._threads.setdefault(thread_id, asyncio.Queue())

    def unsubscribe(self, thread_id: str) -> None:
        self._threads.pop(thread_id, None)

    async def stop(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            with contextlib.suppress(Exception):
                assert self._proc.stdin is not None
                self._proc.stdin.close()
            try:
                await asyncio.wait_for(self._proc.wait(), 5)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)


class _CodexConnection:
    """The sideband shape the voice service reads: Live-style JSON events."""

    def __init__(self, server: _AppServer, thread_id: str, queue: asyncio.Queue) -> None:
        self._server, self._thread, self._queue = server, thread_id, queue
        self._stopped = False
        self._watchdog = asyncio.create_task(self._watch_usage())

    async def _watch_usage(self) -> None:
        while not self._stopped:
            await asyncio.sleep(USAGE_CHECK_SECONDS)
            try:
                problem = await self._server.usage_problem()
            except Exception:
                problem = "usage could not be read"
            if problem and not self._stopped:
                logger.warning("Codex voice stopped by the usage guard: {}", problem)
                await self._stop()

    def __aiter__(self) -> _CodexConnection:
        return self

    async def __anext__(self) -> str:
        while True:
            message = await self._queue.get()
            if message.get("method") in ("thread/realtime/error", "thread/realtime/closed"):
                params = message.get("params") or {}
                logger.warning(
                    "Codex realtime {}: {}",
                    message.get("method").rsplit("/", 1)[-1],
                    str(params.get("message") or params.get("reason"))[:500],
                )
            event = _translate(message)
            if event is not None:
                return json.dumps(event)

    async def send(self, frame: str) -> None:
        event = json.loads(frame)
        if event.get("type") == "session.close":
            await self._stop()
        else:
            logger.debug("Codex voice prototype ignores client event", type=event.get("type"))

    async def _stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            with contextlib.suppress(Exception):
                await self._server.request("thread/realtime/stop", {"threadId": self._thread})

    async def close(self) -> None:
        await self._stop()
        self._watchdog.cancel()
        self._server.unsubscribe(self._thread)


def _translate(message: dict) -> dict | None:
    method, params = message.get("method"), message.get("params") or {}
    if method == "thread/realtime/started":
        return {"type": "session.started"}
    if method == "thread/realtime/transcript/delta":
        kind = "input" if params.get("role") == "user" else "output"
        return {"type": f"session.{kind}_transcript.delta", "delta": params.get("delta")}
    if method == "thread/realtime/closed":
        return {"type": "session.closed", "reason": params.get("reason") or "closed"}
    if method == "thread/realtime/error":
        return {"type": "error", "error": {"code": "codex_realtime_error"}}
    return None


def _initial_items(session: dict) -> list[dict]:
    """The call's seeded history as v3 role/text items."""
    items = []
    for message in session.get("input") or []:
        role = message.get("role")
        text = "".join(part.get("text", "") for part in message.get("content") or [])
        if role in ("user", "assistant") and text:
            items.append({"role": role, "text": text})
    return items


class CodexRealtimeTransport:
    """Same surface as ``LiveTransport``; the ``api_key`` argument is unused."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout
        self._server = _AppServer(timeout)
        self._workdir = tempfile.mkdtemp(prefix="assistant-runtime-codex-voice-")
        self._queues: dict[str, asyncio.Queue] = {}

    async def create(self, api_key: str | None, session: dict, sdp: str) -> tuple[str, str]:
        await self._server.ensure_started()
        problem = await self._server.usage_problem()
        if problem:
            raise VoiceError(
                f"Codex voice refused by the usage guard: {problem}",
                409,
                allocation_status="rejected",
            )
        thread = await self._server.request(
            "thread/start",
            {
                "ephemeral": True,
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "cwd": self._workdir,
            },
        )
        thread_id = thread["thread"]["id"]
        queue = self._server.subscribe(thread_id)
        voice = (session.get("audio") or {}).get("output", {}).get("voice")
        await self._server.request(
            "thread/realtime/start",
            {
                "threadId": thread_id,
                "outputModality": "audio",
                "transport": {"type": "webrtc", "sdp": sdp},
                "prompt": session.get("instructions"),
                "version": _REALTIME_VERSION,
                "includeStartupContext": False,
                "clientManagedHandoffs": True,
                **({"voice": voice} if voice in _CODEX_VOICES else {}),
                **({"initialItems": items} if (items := _initial_items(session)) else {}),
            },
        )
        answer = None
        early: list[dict] = []
        async with asyncio.timeout(self.timeout):
            while answer is None:
                message = await queue.get()
                if message.get("method") == "thread/realtime/sdp":
                    answer = message["params"]["sdp"]
                elif message.get("method") in ("thread/realtime/error", "thread/realtime/closed"):
                    params = message.get("params") or {}
                    logger.warning(
                        "Codex realtime refused the session: {} {}",
                        message.get("method"),
                        str(params.get("message") or params.get("reason"))[:300],
                    )
                    self._server.unsubscribe(thread_id)
                    raise VoiceError(
                        "Codex realtime session was refused", 502, allocation_status="rejected"
                    )
                else:
                    early.append(message)
        for message in early:  # e.g. `started` may precede the SDP
            queue.put_nowait(message)
        return thread_id, answer

    async def attach(self, api_key: str | None, provider_id: str) -> _CodexConnection:
        return _CodexConnection(self._server, provider_id, self._server.subscribe(provider_id))

    async def stop(self) -> None:
        await self._server.stop()
