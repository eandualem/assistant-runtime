"""Codex app-server subprocess ownership and JSON-RPC request correlation."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
from dataclasses import dataclass, field

from loguru import logger

from assistant_runtime.app.voice.exceptions import VoiceError


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
        except Exception as exc:
            logger.warning("Codex app-server reader failed: {}", type(exc).__name__)
        finally:
            # Also when reading fails (an oversized line): nothing reads this
            # server any more, so end it; its requests and calls end with it,
            # and the next call starts a new one.
            if link.proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    link.proc.kill()
            # Drain and reap before releasing waiters or losing the only link to
            # this process; stop() can no longer find it after detachment.
            try:
                await link.proc.communicate()
            finally:
                if self._link is link:
                    self._link = None
                for future in link.pending.values():
                    if not future.done():
                        future.set_exception(VoiceError("Codex app-server exited", 502))
                link.pending.clear()
                for queue in link.threads.values():
                    queue.put_nowait(
                        {
                            "method": "thread/realtime/closed",
                            "params": {"reason": "app_server_exit"},
                        }
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
                with contextlib.suppress(ProcessLookupError):
                    link.proc.kill()
                await link.proc.wait()
        if link.reader is not None:
            # Process exit delivers EOF; let the reader finish draining and
            # notifying its waiters instead of interrupting its finalization.
            await asyncio.gather(link.reader, return_exceptions=True)
