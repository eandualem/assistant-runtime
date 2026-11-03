"""Companion tests for the bounded database connect timeout (issue #220).

A Postgres host that drops packets instead of refusing connections must not
make every call wait asyncpg's default 60-second connect timeout. The runtime's
engine is created with a bounded ``connect_args`` timeout, configured through
``DatabaseConfig.connect_timeout`` (``DATABASE__CONNECT_TIMEOUT``), so a
request waits that long before its ``503`` and a startup probe falls back to
memory just as quickly.

``_MISSING_TIMEOUT`` is the regression contract: it is the failure message the
unpatched runtime prints, because ``DatabaseService.start`` then calls
``create_async_engine`` without a ``connect_args`` timeout at all.
"""

from __future__ import annotations

import asyncio
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from assistant_runtime.config import AppSettings
from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.interface import DatabaseService

_ENGINE = "assistant_runtime.services.database.interface.create_async_engine"

# Printed verbatim by the regression test when the engine is created without a
# bounded connect timeout. The host's base run greps for exactly this line.
_MISSING_TIMEOUT = "connect_args timeout missing"


def _make_mock_engine(*, connect_ok: bool = True) -> tuple[MagicMock, AsyncMock]:
    """Create a mock engine whose begin() answers (or not) like a real probe."""
    mock_conn = AsyncMock()
    mock_engine = MagicMock()
    mock_begin_cm = MagicMock()
    if connect_ok:
        mock_begin_cm.__aenter__ = AsyncMock(return_value=mock_conn)
    else:
        mock_begin_cm.__aenter__ = AsyncMock(
            side_effect=ConnectionRefusedError("Connection refused")
        )
    mock_begin_cm.__aexit__ = AsyncMock(return_value=False)
    mock_engine.begin.return_value = mock_begin_cm
    mock_engine.dispose = AsyncMock()
    return mock_engine, mock_conn


class TestStartBoundsTheConnectTimeout:
    """Regression: the runtime's engine carries a bounded connect timeout."""

    async def test_start_passes_a_bounded_connect_args_timeout(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)
        mock_engine, _ = _make_mock_engine()

        with patch(_ENGINE, return_value=mock_engine) as mock_create:
            await service.start()

        kwargs = mock_create.call_args.kwargs
        # Unpatched code calls create_async_engine without connect_args at all,
        # so this is the assertion the defect fails on.
        assert "connect_args" in kwargs, _MISSING_TIMEOUT
        assert kwargs["connect_args"] == {"timeout": config.connect_timeout}
        assert kwargs["connect_args"]["timeout"] != 60
        assert service.healthy is True

    async def test_start_uses_the_configured_timeout(self):
        service = DatabaseService(config=DatabaseConfig(connect_timeout=2.5))
        mock_engine, _ = _make_mock_engine()

        with patch(_ENGINE, return_value=mock_engine) as mock_create:
            await service.start()

        assert mock_create.call_args.kwargs["connect_args"]["timeout"] == 2.5

    async def test_start_keeps_the_other_engine_settings(self):
        """The bounded timeout is added; pooling and echo stay as configured."""
        config = DatabaseConfig(pool_size=7, pool_overflow=3, echo=True, connect_timeout=1.5)
        service = DatabaseService(config=config)
        mock_engine, _ = _make_mock_engine()

        with patch(_ENGINE, return_value=mock_engine) as mock_create:
            await service.start()

        kwargs = mock_create.call_args.kwargs
        assert kwargs["pool_pre_ping"] is True
        assert kwargs["pool_size"] == 7
        assert kwargs["max_overflow"] == 3
        assert kwargs["echo"] is True
        assert kwargs["connect_args"] == {"timeout": 1.5}

    async def test_unreachable_database_still_degrades_instead_of_waiting(self):
        service = DatabaseService(config=DatabaseConfig(connect_timeout=1.0))
        mock_engine, _ = _make_mock_engine(connect_ok=False)

        with patch(_ENGINE, return_value=mock_engine):
            await service.start()

        assert service._started is True
        assert service.healthy is False

    async def test_required_database_still_fails_fast(self):
        from assistant_runtime.services.database.exceptions import DatabaseUnavailableError

        service = DatabaseService(config=DatabaseConfig(required=True, connect_timeout=1.0))
        mock_engine, _ = _make_mock_engine(connect_ok=False)

        with (
            patch(_ENGINE, return_value=mock_engine),
            pytest.raises(DatabaseUnavailableError, match="DATABASE__REQUIRED"),
        ):
            await service.start()

    async def test_concurrent_starts_keep_their_own_timeouts(self):
        configs = [DatabaseConfig(connect_timeout=t) for t in (1.0, 2.5, 5.0)]
        services = [DatabaseService(config=c) for c in configs]
        seen: list[float] = []

        def fake_create(url, **kwargs):
            seen.append(kwargs["connect_args"]["timeout"])
            return _make_mock_engine()[0]

        with patch(_ENGINE, side_effect=fake_create):
            await asyncio.gather(*(service.start() for service in services))

        # Every service bound its own value; no cross-contamination.
        assert sorted(seen) == [1.0, 2.5, 5.0]
        for service in services:
            assert service.healthy is True


class TestConnectTimeoutConfig:
    """Configuration: default, override, and null/empty/boundary validation."""

    def test_default_is_bounded_below_sixty(self):
        config = DatabaseConfig()
        assert config.connect_timeout == 5.0
        assert config.connect_timeout < 60

    def test_custom_value(self):
        assert DatabaseConfig(connect_timeout=12.5).connect_timeout == 12.5

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("DATABASE__CONNECT_TIMEOUT", "2.5")
        assert AppSettings().database.connect_timeout == 2.5

    def test_null_is_rejected(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(connect_timeout=None)

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(connect_timeout="")

    def test_non_numeric_is_rejected(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(connect_timeout="not-a-timeout")

    def test_zero_is_rejected(self):
        """asyncpg reads 0 as an immediate timeout, so the runtime refuses it."""
        with pytest.raises(ValidationError):
            DatabaseConfig(connect_timeout=0)

    def test_negative_is_rejected(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(connect_timeout=-1)

    def test_smallest_positive_timeout_is_accepted(self):
        assert DatabaseConfig(connect_timeout=0.001).connect_timeout == 0.001

    def test_integer_timeout_is_accepted(self):
        assert DatabaseConfig(connect_timeout=30).connect_timeout == 30.0


class TestSilentPostgresIsBounded:
    """End to end: a packet-dropping server cannot hold startup for a minute.

    A loopback listener accepts the TCP connection and then never answers, which
    is what a host that vanished, a network partition or a firewall that drops
    traffic looks like to asyncpg. The runtime's own engine settings must give
    up after ``connect_timeout`` instead of waiting out asyncpg's 60-second
    default.
    """

    _WAIT_CEILING = 10.0

    async def test_start_against_a_silent_server_is_bounded(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", 0))
        except OSError:  # pragma: no cover - a sandbox without loopback sockets
            listener.close()
            pytest.skip("no loopback socket available")
        listener.listen(8)
        listener.setblocking(False)
        accepted: list[socket.socket] = []

        async def accept_forever() -> None:
            loop = asyncio.get_running_loop()
            while True:
                connection, _ = await loop.sock_accept(listener)
                accepted.append(connection)  # accepted, never answered

        acceptor = asyncio.create_task(accept_forever())
        config = DatabaseConfig(connect_timeout=1.0).model_copy(
            update={"host": "127.0.0.1", "port": listener.getsockname()[1]}
        )
        service = DatabaseService(config=config)
        started = time.monotonic()
        unbounded = False
        try:
            try:
                async with asyncio.timeout(self._WAIT_CEILING):
                    await service.start()
            except TimeoutError:
                unbounded = True
            elapsed = time.monotonic() - started
            healthy = service.healthy
        finally:
            acceptor.cancel()
            listener.close()
            for connection in accepted:
                connection.close()
            await service.stop()

        if unbounded:
            pytest.fail(_MISSING_TIMEOUT)
        assert elapsed < self._WAIT_CEILING
        assert healthy is False
