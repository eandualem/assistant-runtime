"""Local agent sessions as tmux sessions: the helpers the backbone and Claude Code providers share."""

from __future__ import annotations

import asyncio
import re

SESSION_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$")
MAX_SESSION_NAME_LENGTH = 64


async def _run_command(args: list[str], timeout: float = 10.0) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            (stdout_bytes or b"").decode().strip(),
            (stderr_bytes or b"").decode().strip(),
        )
    except FileNotFoundError:
        return (127, "", f"Command not found: {args[0]}")
    except TimeoutError:
        proc.kill()
        return (1, "", f"Command timed out after {timeout}s")


def _validate_session_name(session_name: str) -> str | None:
    """Validate a tmux session name. Returns error message or None if valid."""
    if not session_name:
        return "Session name cannot be empty"
    if len(session_name) > MAX_SESSION_NAME_LENGTH:
        return f"Session name too long (max {MAX_SESSION_NAME_LENGTH} chars)"
    if not SESSION_NAME_PATTERN.match(session_name):
        return "Session name must be alphanumeric with hyphens, starting with alphanumeric"
    return None
