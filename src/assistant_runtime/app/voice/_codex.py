"""Voice through the local Codex CLI's realtime interface, on its ChatGPT login.

The runtime starts the official ``codex app-server`` and never reads or holds
credentials: the CLI authenticates with its own login. API-key variables are
removed from its environment, so the provider cannot fall back to API billing.
Each call gets an ephemeral read-only thread and a realtime v3 WebRTC session;
the browser's offer is forwarded unchanged and audio flows directly between the
browser and the provider.

Every realtime delegation starts a turn of the Codex agent behind the thread.
Handoffs are client-managed, so that agent's output never reaches the call, and
any turn it starts on the call's thread is interrupted: the runtime answers.

``thread/realtime/*`` is experimental in the CLI, and the app-server ignores
start parameters it does not know. Before the first call the installed CLI's
protocol schema is checked for everything this transport relies on.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from assistant_runtime.app.voice.exceptions import VoiceError

REALTIME_VERSION = "v3"
# What this transport calls and listens for; a CLI without them is incompatible.
_REQUIRED_METHODS = {
    "ClientRequest.json": {
        "account/read",
        "account/rateLimits/read",
        "thread/start",
        "thread/realtime/start",
        "thread/realtime/stop",
        "thread/realtime/appendSpeech",
        "thread/realtime/appendText",
        "turn/interrupt",
    },
    "ServerNotification.json": {
        "thread/realtime/sdp",
        "thread/realtime/started",
        "thread/realtime/closed",
        "thread/realtime/error",
        "thread/realtime/transcript/delta",
        "thread/realtime/transcript/done",
        "thread/realtime/itemAdded",
        "turn/started",
    },
}
_REQUIRED_THREAD_PARAMS = {"approvalPolicy", "cwd", "developerInstructions", "ephemeral", "sandbox"}
_REQUIRED_START_PARAMS = {
    "clientManagedHandoffs",
    "includeStartupContext",
    "initialItems",
    "outputModality",
    "prompt",
    "transport",
    "version",
    "voice",
}


def _methods(schema: dict) -> set[str]:
    methods = set()
    for variant in schema.get("oneOf", []):
        method = (variant.get("properties") or {}).get("method") or {}
        # One unusual variant must not hide the others.
        names = method.get("enum") or ([method["const"]] if "const" in method else [])
        methods.update(name for name in names if isinstance(name, str))
    return methods


def missing_from_schema(directory: Path) -> list[str]:
    """Names this transport needs that the generated app-server schema lacks."""
    missing = []
    for file, needed in _REQUIRED_METHODS.items():
        try:
            present = _methods(json.loads((directory / file).read_text()))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            present = set()
        missing += sorted(needed - present)
    try:
        start = json.loads((directory / "v2" / "ThreadRealtimeStartParams.json").read_text())
    except (OSError, ValueError):
        start = {}
    try:
        params = set(start.get("properties", {}))
    except (AttributeError, TypeError):
        params = set()
    try:
        versions = list(start["definitions"]["RealtimeConversationVersion"]["enum"])
    except (KeyError, TypeError):
        versions = []
    missing += [f"thread/realtime/start.{name}" for name in sorted(_REQUIRED_START_PARAMS - params)]
    if REALTIME_VERSION not in versions:
        missing.append(f"thread/realtime/start.version={REALTIME_VERSION}")
    try:
        started = json.loads(
            (directory / "v2" / "ThreadRealtimeStartedNotification.json").read_text()
        )
        started_fields = set(started.get("properties", {}))
    except (OSError, ValueError, TypeError, AttributeError):
        started_fields = set()
    if "version" not in started_fields:  # the session's version is checked on start
        missing.append("thread/realtime/started.version")
    try:
        limits = json.loads((directory / "v2" / "GetAccountRateLimitsResponse.json").read_text())
        snapshot = set(limits["definitions"]["RateLimitSnapshot"]["properties"])
    except (OSError, ValueError, KeyError, TypeError):
        snapshot = set()
    # The usage guard reads these; an absent credits value is trusted only
    # because the protocol defines the field.
    missing += [
        f"account/rateLimits/read.{name}"
        for name in sorted({"credits", "spendControlReached"} - snapshot)
    ]
    for file, method, needed in (
        ("ThreadStartParams.json", "thread/start", _REQUIRED_THREAD_PARAMS),
        ("ThreadRealtimeAppendTextParams.json", "thread/realtime/appendText", {"role"}),
    ):
        try:
            fields = set(json.loads((directory / "v2" / file).read_text()).get("properties", {}))
        except (OSError, ValueError, AttributeError, TypeError):
            fields = set()
        missing += [f"{method}.{name}" for name in sorted(needed - fields)]
    return missing


def usage_report(limits: dict, ceiling_percent: int) -> dict:
    """Normalise ``account/rateLimits/read`` and decide whether a call may use it.

    The guard never lets a call spend purchased credits: it refuses while any
    credit balance could be charged, when spend control or a reached limit is
    reported, when included usage is not allowed, or when a usage window is at
    the configured ceiling.
    """
    snapshots = limits.get("rateLimitsByLimitId") or {}
    if not isinstance(snapshots, dict):
        snapshots = {}
    if not snapshots and limits.get("rateLimits"):
        snapshots = {"default": limits["rateLimits"]}
    windows, reasons = [], []
    has_credits = spend_control = False
    if not any(isinstance(snapshot, dict) for snapshot in snapshots.values()):
        # A shape this runtime does not recognise must not read as "allowed".
        reasons.append("usage_unreadable")
    for limit_id, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            continue
        balance = snapshot.get("credits")
        if balance is None:
            balance = {}  # optional in the CLI's schema: no credit information
        elif not isinstance(balance, dict):
            reasons.append("usage_unreadable")
            balance = {}
        if (
            balance.get("hasCredits")
            or balance.get("unlimited")
            or not _is_zero(balance.get("balance"))
        ):
            has_credits = True
        if snapshot.get("spendControlReached"):
            spend_control = True
        if snapshot.get("rateLimitReachedType"):
            reasons.append("usage_limit_reached")
        for name in ("primary", "secondary"):
            window = snapshot.get(name)
            if not isinstance(window, dict) or window.get("usedPercent") is None:
                continue
            if not isinstance(window["usedPercent"], int | float):
                reasons.append("usage_unreadable")
                continue
            windows.append(
                {
                    "limit": limit_id,
                    "window": name,
                    "used_percent": window["usedPercent"],
                    "window_minutes": window.get("windowDurationMins"),
                    "resets_at": window.get("resetsAt"),
                }
            )
            if window["usedPercent"] >= ceiling_percent:
                reasons.append("usage_window_limit")
    if has_credits:
        reasons.insert(0, "credits_available")
    if spend_control:
        reasons.insert(0, "spend_control_reached")
    if limits.get("ordinaryUsageAllowed") is False:
        reasons.insert(0, "usage_not_allowed")
    reason = reasons[0] if reasons else None
    return {
        "allowed": reason is None,
        "reason": reason,
        "ceiling_percent": ceiling_percent,
        "windows": windows,
        "credits_available": has_credits,
        "spend_control_reached": spend_control,
        "ordinary_usage_allowed": limits.get("ordinaryUsageAllowed"),
    }


# For the Codex agent behind a call's thread, which never answers the call.
_AGENT_INSTRUCTIONS = (
    "This thread only carries a voice call, and another system answers every request. "
    "Do not run commands, read files or call tools. End every turn at once without output."
)
# Consecutive unreadable usage checks that end a call.
_UNREADABLE_CHECKS = 3
# Closures the runtime synthesises when it can no longer see the session.
_LOST = frozenset({"app_server_exit", "stop_failed"})


class _CommandFailedError(RuntimeError):
    """The Codex CLI ran and exited with an error."""


def _is_zero(balance: Any) -> bool:
    """A missing or zero balance; anything unreadable counts as spendable."""
    if balance is None:
        return True
    try:
        return float(balance) == 0
    except (TypeError, ValueError):
        return False


def _error_code(message: str) -> str:
    # The app-server reports provider errors as free text; expose a code only.
    return "usage_limit_reached" if "usage limit" in message.lower() else "codex_realtime_error"


@dataclass
class _Link:
    """One spawned app-server, with the requests and calls that depend on it."""

    proc: asyncio.subprocess.Process
    pending: dict[int, asyncio.Future] = field(default_factory=dict)
    threads: dict[str, asyncio.Queue] = field(default_factory=dict)
    reader: asyncio.Task | None = None


class _AppServer:
    """Newline-delimited JSON-RPC over ``codex app-server --stdio``."""

    def __init__(self, command: str, timeout: float) -> None:
        self.command = command
        self.timeout = timeout
        self._link: _Link | None = None
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self._link is not None and self._link.proc.returncode is None

    async def ensure_started(self) -> None:
        async with self._lock:
            if self.running:
                return
            env = {
                k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY")
            }
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.command,
                    "app-server",
                    "--stdio",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    env=env,
                    limit=2**24,
                )
            except OSError as exc:
                raise VoiceError(
                    "Codex voice needs the Codex CLI; set VOICE__CODEX_COMMAND",
                    503,
                    allocation_status="rejected",
                ) from exc
            link = self._link = _Link(proc)
            link.reader = asyncio.create_task(self._read(link))
            try:
                await self.request(
                    "initialize",
                    {
                        "clientInfo": {"name": "assistant-runtime", "version": "voice"},
                        # thread/realtime/* is experimental; without this the
                        # app-server drops its notifications silently.
                        "capabilities": {"experimentalApi": True},
                    },
                )
                await self._write(link, {"jsonrpc": "2.0", "method": "initialized"})
                await self.require_chatgpt()
            except BaseException as exc:
                # A half-initialised server, or one on another login, must not
                # serve a later call: the next call starts and checks a new one.
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                self._link = None
                if not isinstance(exc, asyncio.CancelledError):
                    await asyncio.wait([link.reader], timeout=5)
                raise

    async def require_chatgpt(self) -> None:
        """Refuse unless the CLI is signed in with ChatGPT (never an API key)."""
        account = await self.request("account/read", {})
        if (account.get("account") or {}).get("type") != "chatgpt":
            raise VoiceError(
                "Codex voice requires the Codex CLI to be signed in with ChatGPT; "
                "run `codex login`",
                503,
                allocation_status="rejected",
            )

    @staticmethod
    async def _write(link: _Link, message: dict) -> None:
        assert link.proc.stdin is not None
        link.proc.stdin.write((json.dumps(message) + "\n").encode())
        await link.proc.stdin.drain()

    async def _read(self, link: _Link) -> None:
        assert link.proc.stdout is not None
        try:
            while line := await link.proc.stdout.readline():
                try:
                    message = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(message, dict):
                    continue
                ident = message.get("id")
                if "method" in message and ident is not None:
                    # A server request (approval, tool call): this client serves none.
                    with contextlib.suppress(Exception):
                        await self._write(
                            link,
                            {
                                "jsonrpc": "2.0",
                                "id": ident,
                                "error": {"code": -32601, "message": "unsupported by this client"},
                            },
                        )
                    continue
                if ident in link.pending and ("result" in message or "error" in message):
                    future = link.pending.pop(ident)
                    if not future.done():
                        future.set_result(message)
                    continue
                thread = (message.get("params") or {}).get("threadId")
                if thread in link.threads:
                    link.threads[thread].put_nowait(message)
        finally:
            # Also when reading fails (an oversized line): nothing reads this
            # server any more, so end it; its requests and calls end with it,
            # and the next call starts a new one.
            if self._link is link:
                self._link = None
            if link.proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    link.proc.kill()
            for future in link.pending.values():
                if not future.done():
                    future.set_exception(VoiceError("Codex app-server exited", 502))
            link.pending.clear()
            for queue in link.threads.values():
                queue.put_nowait(
                    {"method": "thread/realtime/closed", "params": {"reason": "app_server_exit"}}
                )

    async def request(self, method: str, params: dict) -> dict:
        link = self._link
        if link is None:
            raise VoiceError("Codex app-server is not running", 502)
        ident = next(self._ids)
        future = asyncio.get_running_loop().create_future()
        link.pending[ident] = future
        try:
            await self._write(
                link, {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
            )
            message = await asyncio.wait_for(future, self.timeout)
        except (OSError, TimeoutError) as exc:
            raise VoiceError(f"Codex {method} failed", 502) from exc
        finally:
            link.pending.pop(ident, None)
        if "error" in message:
            # Error text can carry account or prompt detail; keep it out of responses.
            logger.warning("Codex app-server {} failed: {}", method, str(message["error"])[:300])
            raise VoiceError(f"Codex {method} failed", 502)
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def subscribe(self, thread_id: str) -> asyncio.Queue:
        if self._link is None:
            raise VoiceError("Codex app-server is not running", 502)
        return self._link.threads.setdefault(thread_id, asyncio.Queue())

    def unsubscribe(self, thread_id: str) -> None:
        if self._link is not None:
            self._link.threads.pop(thread_id, None)

    async def stop(self) -> None:
        link, self._link = self._link, None
        if link is None:
            return
        if link.proc.returncode is None:
            with contextlib.suppress(Exception):
                assert link.proc.stdin is not None
                link.proc.stdin.close()
            try:
                await asyncio.wait_for(link.proc.wait(), 5)
            except TimeoutError:
                link.proc.kill()
                await link.proc.wait()
        if link.reader is not None:
            link.reader.cancel()
            await asyncio.gather(link.reader, return_exceptions=True)


class _CodexConnection:
    """The sideband the voice service reads, as Live-style JSON events."""

    def __init__(
        self, server: _AppServer, thread_id: str, queue: asyncio.Queue, transport: CodexTransport
    ) -> None:
        self._server, self._thread, self._queue = server, thread_id, queue
        self._transport = transport
        self._stopping = False
        self._stop_reason: str | None = None
        self._stop_task: asyncio.Task | None = None
        self._closed = False  # the provider session has ended
        self._started_at: float | None = None
        self._tasks: set[asyncio.Task] = set()
        self._watchdog = asyncio.create_task(self._watch_usage())

    async def _watch_usage(self) -> None:
        failures = 0
        while not self._stopping:
            await asyncio.sleep(self._transport.usage_check_seconds)
            try:
                # Never start a server for a call whose server has gone.
                reason = (await self._transport.usage(start=False))["reason"]
            except Exception:
                reason = "usage_unreadable"
            if reason == "usage_unreadable":
                # One bad read is not a reason to end a call; several in a row are.
                failures += 1
                if failures < _UNREADABLE_CHECKS:
                    reason = None
            else:
                failures = 0
            if reason and not self._stopping:
                logger.warning("Codex voice stopped by the usage guard: {}", reason)
                # Queued ahead of the closed notification the stop produces.
                self._queue.put_nowait({"method": "usage_guard", "params": {"code": reason}})
                await asyncio.shield(self._stop("usage_guard"))

    def __aiter__(self) -> _CodexConnection:
        return self

    async def __anext__(self) -> str:
        while True:
            message = await self._queue.get()
            if message.get("method") == "thread/realtime/closed" and (
                (message.get("params") or {}).get("reason") in _LOST
            ):
                # Nothing confirms the provider session ended: report interruption.
                raise ConnectionError("Codex app-server connection lost")
            event = self._translate(message)
            if event is not None:
                return json.dumps(event)

    def _translate(self, message: dict) -> dict | None:
        method, params = message.get("method"), message.get("params") or {}
        if method == "usage_guard":
            return {"type": "error", "error": {"code": params.get("code")}}
        if method == "speech_queued":
            return {"type": "session.commentary.appended", "client_event_id": params.get("id")}
        if method == "speech_failed":
            return {
                "type": "error",
                "error": {"code": "codex_realtime_error", "client_event_id": params.get("id")},
            }
        if method == "thread/realtime/itemAdded":
            item = params.get("item") or {}
            if item.get("type") != "handoff_request" or not item.get("handoff_id"):
                # v3 adds only delegations here; anything else may be protocol drift.
                logger.warning("Codex voice ignored a realtime item of type {}", item.get("type"))
                return None
            return {
                "type": "session.delegation.created",
                "delegation": {
                    "id": item["handoff_id"],
                    "target": "client",
                    "input": item.get("input_transcript"),
                },
            }
        if method == "turn/started":
            # The backing Codex agent must not act on the call's behalf.
            turn = (params.get("turn") or {}).get("id")
            if isinstance(turn, str):
                self._spawn(self._interrupt(turn))
            return None
        if method == "thread/realtime/started":
            self._started_at = time.monotonic()
            if params.get("version") != REALTIME_VERSION:
                self._stop("codex_version_mismatch")
                return {"type": "error", "error": {"code": "codex_version_mismatch"}}
            return {"type": "session.started"}
        if method == "thread/realtime/transcript/delta":
            kind = "input" if params.get("role") == "user" else "output"
            return {"type": f"session.{kind}_transcript.delta", "delta": params.get("delta")}
        if method == "thread/realtime/transcript/done":
            return {
                "type": "session.transcript.done",
                "role": "user" if params.get("role") == "user" else "assistant",
                "text": params.get("text"),
            }
        if method == "thread/realtime/error":
            message_text = str(params.get("message") or "")
            logger.warning("Codex realtime error: {}", message_text[:300])
            return {"type": "error", "error": {"code": _error_code(message_text)}}
        if method == "thread/realtime/closed":
            self._closed = True
            reason = params.get("reason")
            duration = round(time.monotonic() - self._started_at, 3) if self._started_at else None
            return {
                "type": "session.closed",
                # "requested" is the runtime's own stop: keep the service's reason.
                "reason": self._stop_reason or (None if reason == "requested" else reason),
                # The provider reports no realtime usage; the measured duration is all there is.
                "usage": {"duration_seconds": duration} if duration is not None else {},
            }
        return None

    async def _interrupt(self, turn: str) -> None:
        try:
            await self._server.request("turn/interrupt", {"threadId": self._thread, "turnId": turn})
        except Exception:
            logger.warning("Codex voice could not interrupt the Codex agent's turn")

    def _spawn(self, awaitable: Any) -> None:
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(lambda done: done.cancelled() or done.exception())

    async def send(self, frame: str) -> None:
        event = json.loads(frame)
        if event.get("type") == "session.close":
            # Not awaited: the service waits for the closed notification, and a
            # slow stop must not turn into a send timeout. close() awaits it.
            self._stop(None)
        elif event.get("type") == "session.commentary.append":
            # What Codex's own client does with a delegated result. Speech is
            # session-level context: the protocol names no delegation.
            self._spawn(self._speak(event.get("event_id"), str(event.get("content") or "")))
        else:
            logger.debug("Codex voice ignores client event", type=event.get("type"))

    async def append_fact(self, text: str, *, speak: bool) -> None:
        """Deliver a host fact: quiet context, or speech now. Raises when refused.

        On realtime v3 the app-server sends text without a channel and the
        provider treats that as speakable, so quiet facts rely on the call's
        instructions, not on the protocol.
        """
        if self._stopping:
            raise VoiceError("Voice call is closing")
        if speak:
            await self._server.request(
                "thread/realtime/appendSpeech", {"threadId": self._thread, "text": text}
            )
        else:
            await self._server.request(
                "thread/realtime/appendText",
                {"threadId": self._thread, "text": text, "role": "developer"},
            )

    async def _speak(self, command_id: Any, text: str) -> None:
        try:
            await self._server.request(
                "thread/realtime/appendSpeech", {"threadId": self._thread, "text": text}
            )
        except Exception:
            self._queue.put_nowait({"method": "speech_failed", "params": {"id": command_id}})
        else:
            # Queued for the provider, not proof that it was spoken.
            self._queue.put_nowait({"method": "speech_queued", "params": {"id": command_id}})

    def _stop(self, reason: str | None) -> asyncio.Task:
        """Stop the session once; every caller shares the same stop."""
        if self._stop_task is None:
            self._stopping = True
            self._stop_reason = reason
            self._stop_task = asyncio.create_task(self._request_stop())
        return self._stop_task

    async def _request_stop(self) -> None:
        try:
            await self._server.request("thread/realtime/stop", {"threadId": self._thread})
        except Exception:
            # No closed notification will follow a failed stop.
            self._queue.put_nowait(
                {"method": "thread/realtime/closed", "params": {"reason": "stop_failed"}}
            )

    async def close(self) -> None:
        # Bounded: a hung app-server must not hold the call's session reservation.
        if not self._closed or self._stop_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.shield(self._stop(None)), self._transport.close_timeout
                )
        self._stopping = True
        self._watchdog.cancel()
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(self._watchdog, *self._tasks, return_exceptions=True)
        self._server.unsubscribe(self._thread)
        # Let the app-server release the call's thread; best effort.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self._server.request("thread/unsubscribe", {"threadId": self._thread}),
                self._transport.close_timeout,
            )


class CodexTransport:
    """Same surface as ``LiveTransport``; the ``api_key`` arguments are unused."""

    def __init__(
        self,
        command: str,
        timeout: float,
        *,
        usage_ceiling_percent: int,
        usage_check_seconds: float,
        close_timeout: float,
    ) -> None:
        self.command = command
        self.timeout = timeout
        # Bounds each of a closing call's stop and thread release, so a hung
        # app-server cannot hold the call's session reservation.
        self.close_timeout = close_timeout
        self.usage_ceiling_percent = usage_ceiling_percent
        self.usage_check_seconds = usage_check_seconds
        self._server = _AppServer(command, timeout)
        self._workdir: tempfile.TemporaryDirectory | None = None  # threads' empty cwd
        self._compatibility: dict | None = None
        self._check_lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()
        self._created: dict[str, asyncio.Queue] = {}  # created, not yet attached

    def checked(self) -> dict | None:
        """The completed protocol check, without running one."""
        return self._compatibility

    async def compatibility(self) -> dict:
        """Check the installed CLI's protocol, offline (no session, no login).

        Only a completed check is kept; a CLI that could not be run is checked
        again on the next call.
        """
        async with self._check_lock:
            if self._compatibility is not None:
                return self._compatibility
            result = await self._check()
            if result["missing"] != ["codex_cli"]:
                self._compatibility = result
            return result

    async def _output(self, *args: str, timeout: float) -> bytes:
        proc = await asyncio.create_subprocess_exec(
            self.command, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode:
            raise _CommandFailedError(f"codex {args[0]} exited with {proc.returncode}")
        return stdout

    async def _check(self) -> dict:
        version = None
        try:
            with tempfile.TemporaryDirectory(prefix="assistant-runtime-codex-schema-") as out:
                stdout = await self._output("--version", timeout=30)
                version = stdout.decode(errors="ignore").strip().rsplit(" ", 1)[-1] or None
                try:
                    await self._output(
                        "app-server",
                        "generate-json-schema",
                        "--experimental",
                        "--out",
                        out,
                        timeout=60,
                    )
                except _CommandFailedError:
                    # The CLI runs but cannot describe its protocol: too old, not missing.
                    missing = ["app-server generate-json-schema"]
                else:
                    missing = missing_from_schema(Path(out))
        except Exception as exc:
            logger.warning("Codex CLI check could not run: {}", type(exc).__name__)
            return {"version": version, "compatible": False, "missing": ["codex_cli"]}
        return {"version": version, "compatible": not missing, "missing": missing}

    async def usage(self, *, start: bool = True) -> dict:
        if start:
            await self._server.ensure_started()
        elif not self._server.running:
            raise VoiceError("Codex app-server is not running", 502)
        limits = await self._server.request("account/rateLimits/read", {})
        return usage_report(limits, self.usage_ceiling_percent)

    async def create(self, api_key: str | None, session: dict, sdp: str) -> tuple[str, str]:
        check = await self.compatibility()
        if check["missing"] == ["codex_cli"]:
            raise VoiceError(
                "Codex voice needs the Codex CLI; install it or set VOICE__CODEX_COMMAND",
                503,
                allocation_status="rejected",
                reason="codex_unavailable",
            )
        if not check["compatible"]:
            raise VoiceError(
                "The installed Codex CLI lacks the realtime interface this runtime needs: "
                + ", ".join(check["missing"]),
                503,
                allocation_status="rejected",
                reason="codex_incompatible",
            )
        report = await self.usage()  # starts the app-server when needed
        # The CLI's login can change while its server runs: check it for every call.
        await self._server.require_chatgpt()
        if report["reason"]:
            raise VoiceError(
                f"Codex voice refused by the usage guard: {report['reason']}",
                409,
                allocation_status="rejected",
                reason=report["reason"],
            )
        if self._workdir is None:
            self._workdir = tempfile.TemporaryDirectory(prefix="assistant-runtime-codex-voice-")
        thread = await self._server.request(
            "thread/start",
            {
                "ephemeral": True,
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "cwd": self._workdir.name,
                # Its turns are interrupted as they start; until then it must not act.
                "developerInstructions": _AGENT_INSTRUCTIONS,
            },
        )
        thread_id = (thread.get("thread") or {}).get("id")
        if not isinstance(thread_id, str) or not thread_id:
            raise VoiceError(
                "Codex thread/start returned no thread", 502, allocation_status="rejected"
            )
        queue = self._server.subscribe(thread_id)
        try:
            await self._server.request(
                "thread/realtime/start",
                {
                    "threadId": thread_id,
                    "outputModality": "audio",
                    "transport": {"type": "webrtc", "sdp": sdp},
                    "prompt": session["instructions"],
                    "version": REALTIME_VERSION,
                    "includeStartupContext": False,
                    "clientManagedHandoffs": True,
                    "voice": session["voice"],
                    **({"initialItems": session["history"]} if session["history"] else {}),
                },
            )
            early: list[dict] = []
            async with asyncio.timeout(self.timeout):
                while True:
                    message = await queue.get()
                    method = message.get("method")
                    if method == "thread/realtime/sdp":
                        answer = (message.get("params") or {}).get("sdp")
                        break
                    if method == "thread/realtime/closed" and (
                        (message.get("params") or {}).get("reason") in _LOST
                    ):
                        # The server went away: nothing says the session was not allocated.
                        raise VoiceError(
                            "Codex app-server exited while starting the session",
                            502,
                            allocation_status="unknown",
                        )
                    if method in ("thread/realtime/error", "thread/realtime/closed"):
                        text = str((message.get("params") or {}).get("message") or "")
                        logger.warning("Codex realtime refused the session: {}", text[:300])
                        code = _error_code(text)
                        raise VoiceError(
                            "Codex realtime session was refused",
                            502,
                            allocation_status="rejected",
                            reason=code if code == "usage_limit_reached" else None,
                        )
                    early.append(message)
            if not isinstance(answer, str) or not answer:
                raise VoiceError(
                    "Codex realtime returned no answer", 502, allocation_status="unknown"
                )
        except BaseException:
            # The session may have started even though no answer arrived; stop it
            # in the background so a cancelled request cannot skip the stop.
            self.abandon(thread_id)
            raise
        # Keep arrival order: what came before the answer goes first.
        later = []
        while not queue.empty():
            later.append(queue.get_nowait())
        for message in (*early, *later):
            queue.put_nowait(message)
        self._created[thread_id] = queue
        return thread_id, answer

    def abandon(self, provider_id: str) -> None:
        """Stop a session that was created but never attached to a call."""
        self._created.pop(provider_id, None)
        self._server.unsubscribe(provider_id)
        task = asyncio.create_task(self._stop_quietly(provider_id))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _stop_quietly(self, thread_id: str) -> None:
        for method in ("thread/realtime/stop", "thread/unsubscribe"):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self._server.request(method, {"threadId": thread_id}), self.close_timeout
                )

    async def attach(self, api_key: str | None, provider_id: str) -> _CodexConnection:
        # The queue create() read from, even if its server has since been replaced.
        queue = self._created.pop(provider_id, None) or self._server.subscribe(provider_id)
        return _CodexConnection(self._server, provider_id, queue, self)

    async def stop(self) -> None:
        await asyncio.gather(*self._background, return_exceptions=True)
        await self._server.stop()
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None
