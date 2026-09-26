"""Local CLI processes are reaped before timeout or cancellation is returned."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.services.tools.providers import _local_sessions
from assistant_runtime.services.tools.providers._local_sessions import _run_command


@pytest.mark.parametrize("cancel", [False, True])
async def test_interrupted_command_waits_for_process_cleanup(cancel):
    entered, killed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    process = MagicMock(returncode=None)

    async def communicate():
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        await release.wait()
        process.returncode = -9
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = killed.set
    with patch("asyncio.create_subprocess_exec", return_value=process):
        task = asyncio.create_task(_run_command(["test-cli"], timeout=1 if cancel else 0.01))
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
        await asyncio.wait_for(killed.wait(), 1)
        assert not task.done()
        release.set()
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            code, _, error = await task
            assert code == 1
            assert "timed out" in error
    assert process.returncode == -9


async def test_repeated_cancellation_still_completes_process_cleanup():
    entered, killed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    process = MagicMock(returncode=None)

    async def communicate():
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        await release.wait()
        process.returncode = -9
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = killed.set
    with patch("asyncio.create_subprocess_exec", return_value=process):
        task = asyncio.create_task(_run_command(["test-cli"], timeout=1))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        await asyncio.wait_for(killed.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert process.returncode == -9


async def test_cancellation_during_timeout_cleanup_propagates():
    entered, killed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    process = MagicMock(returncode=None)

    async def communicate():
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        await release.wait()
        process.returncode = -9
        return b"", b""

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = killed.set
    with patch("asyncio.create_subprocess_exec", return_value=process):
        task = asyncio.create_task(_run_command(["test-cli"], timeout=0.01))
        await asyncio.wait_for(killed.wait(), 1)
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert process.returncode == -9


@pytest.mark.parametrize("cancel", [False, True])
async def test_cleanup_is_bounded_when_pipes_never_close(cancel, monkeypatch):
    monkeypatch.setattr(_local_sessions, "CLEANUP_SECONDS", 0.05)
    entered, killed = asyncio.Event(), asyncio.Event()
    process = MagicMock(returncode=None)

    async def communicate():
        # A descendant keeps the pipes open: EOF never arrives.
        entered.set()
        await asyncio.Event().wait()

    process.communicate = AsyncMock(side_effect=communicate)
    process.kill.side_effect = killed.set
    with patch("asyncio.create_subprocess_exec", return_value=process):
        task = asyncio.create_task(_run_command(["test-cli"], timeout=1 if cancel else 0.01))
        await asyncio.wait_for(entered.wait(), 1)
        if cancel:
            task.cancel()
        await asyncio.wait_for(killed.wait(), 1)
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        else:
            code, _, error = await asyncio.wait_for(task, 1)
            assert code == 1
            assert "timed out" in error


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell fixture")
async def test_bounded_cleanup_closes_pipes_held_by_a_descendant(monkeypatch):
    monkeypatch.setattr(_local_sessions, "CLEANUP_SECONDS", 0.2)
    spawned = []
    real_exec = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        proc = await real_exec(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    # The background sleep inherits stdout and stderr and outlives the killed shell.
    code, _, error = await asyncio.wait_for(
        _run_command(["sh", "-c", "sleep 3 & exec sleep 3"], timeout=0.1), 2
    )
    assert code == 1
    assert "timed out" in error
    assert spawned[0]._transport.is_closing()
