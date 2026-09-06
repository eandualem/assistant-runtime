"""``assistant-runtime serve``: run the HTTP + Socket.IO server with uvicorn.

A previous runtime left on the same port (a forgotten terminal, an agent's
instance) is the usual reason ``serve`` fails with "address already in use".
By default ``serve`` recognises such an instance through its ``/health``
endpoint, asks it to stop, waits for the port, then starts; anything else
listening on the port is left alone and reported. ``--no-replace`` disables
the takeover.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPLACE_TIMEOUT_SECONDS = 10.0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the ASGI app. Blocks until the server stops."""
    import uvicorn

    if args.verbose:
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
    if not getattr(args, "no_replace", False) and not replace_previous_instance(
        args.host, args.port
    ):
        return 1
    uvicorn.run(
        "assistant_runtime.main:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_level="info",
    )
    return 0


def replace_previous_instance(
    host: str,
    port: int,
    *,
    timeout: float = REPLACE_TIMEOUT_SECONDS,
    log: Callable[[str], None] = print,
) -> bool:
    """Free ``host:port`` when a previous assistant-runtime holds it.

    Returns True when the port is free (already, or after the previous
    instance stopped) and False when it stays busy: another program owns it,
    the listener's process could not be found, or it did not exit in time.
    """
    if not port_in_use(host, port):
        return True
    if not is_assistant_runtime(host, port):
        log(
            f"serve: {host}:{port} is in use by something that is not an assistant-runtime; "
            "choose another --port or stop that program."
        )
        return False
    pids = listener_pids(host, port)
    if not pids:
        log(
            f"serve: a previous assistant-runtime is listening on {host}:{port} but its process "
            "could not be found (is `lsof` installed?); stop it yourself or use another --port."
        )
        return False
    if not all(_signal(pid, signal.SIGTERM) for pid in pids):
        log(
            f"serve: not allowed to stop the previous assistant-runtime on {host}:{port} "
            f"(pid {pids}); stop it yourself or use another --port."
        )
        return False
    if _wait_until_free(host, port, timeout):
        log(f"serve: replaced the previous assistant-runtime on {host}:{port} (pid {pids})")
        return True
    for pid in pids:
        _signal(pid, signal.SIGKILL)
    if _wait_until_free(host, port, 2.0):
        log(f"serve: replaced an unresponsive assistant-runtime on {host}:{port} (pid {pids})")
        return True
    log(f"serve: {host}:{port} is still in use after stopping pid {pids}; giving up.")
    return False


def port_in_use(host: str, port: int) -> bool:
    """Whether something accepts TCP connections on ``host:port``."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


HEALTH_PROBE_SECONDS = 3.0
_HEALTH_BODY_LIMIT = 64 * 1024


def is_assistant_runtime(host: str, port: int, *, deadline: float = HEALTH_PROBE_SECONDS) -> bool:
    """Whether the listener answers ``/health`` like this runtime (200 or 503 with ``healthy``).

    The whole probe, body read included, is bounded by ``deadline``; the
    socket timeout alone only bounds each individual read.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        body = pool.submit(_read_health, host, port).result(timeout=deadline)
    except FutureTimeout:
        return False
    except Exception:  # noqa: BLE001 - any failure means "not ours"
        return False
    finally:
        # A stuck read finishes on its own socket timeout; do not wait for it here.
        pool.shutdown(wait=False)
    if body is None:
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    return isinstance(payload, dict) and "healthy" in payload


def _read_health(host: str, port: int) -> bytes | None:
    try:
        with urlopen(f"http://{_endpoint(host)}:{port}/health", timeout=2.0) as response:  # noqa: S310
            return response.read(_HEALTH_BODY_LIMIT)
    except HTTPError as error:
        if error.code != 503:
            return None
        return error.read(_HEALTH_BODY_LIMIT)
    except (URLError, OSError, ValueError):
        return None


def _endpoint(host: str) -> str:
    """``host`` as it appears in a URL or lsof address: IPv6 literals in brackets."""
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


_WILDCARD_HOSTS = {"", "0.0.0.0", "::", "*"}


def listener_pids(host: str, port: int) -> list[int]:
    """Process ids listening on exactly ``host:port``, through ``lsof`` when available.

    The address is part of the filter so a different program bound to the
    same port on another interface is never signalled; a wildcard bind
    matches any address.
    """
    if shutil.which("lsof") is None:
        return []
    address = f"-iTCP:{port}" if host in _WILDCARD_HOSTS else f"-iTCP@{_endpoint(host)}:{port}"
    try:
        output = subprocess.run(  # noqa: S603
            ["lsof", "-t", address, "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in output.split() if line.strip().isdigit()]


def _signal(pid: int, sig: signal.Signals) -> bool:
    """Send ``sig``; a vanished process counts as done, a refused one as failure."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _wait_until_free(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_in_use(host, port):
            return True
        time.sleep(0.2)
    return not port_in_use(host, port)
