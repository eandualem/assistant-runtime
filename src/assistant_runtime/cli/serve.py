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
import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable
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
    pids = listener_pids(port)
    if not pids:
        log(
            f"serve: a previous assistant-runtime is listening on {host}:{port} but its process "
            "could not be found (is `lsof` installed?); stop it yourself or use another --port."
        )
        return False
    for pid in pids:
        _signal(pid, signal.SIGTERM)
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


def is_assistant_runtime(host: str, port: int) -> bool:
    """Whether the listener answers ``/health`` like this runtime (200 or 503 with ``healthy``)."""
    try:
        with urlopen(f"http://{host}:{port}/health", timeout=2.0) as response:  # noqa: S310
            body = response.read()
    except HTTPError as error:
        if error.code != 503:
            return False
        body = error.read()
    except (URLError, OSError, ValueError):
        return False
    try:
        return isinstance(json.loads(body), dict) and "healthy" in json.loads(body)
    except ValueError:
        return False


def listener_pids(port: int) -> list[int]:
    """Process ids listening on ``port``, through ``lsof`` when it is available."""
    if shutil.which("lsof") is None:
        return []
    try:
        output = subprocess.run(  # noqa: S603
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(line) for line in output.split() if line.strip().isdigit()]


def _signal(pid: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _wait_until_free(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not port_in_use(host, port):
            return True
        time.sleep(0.2)
    return not port_in_use(host, port)
