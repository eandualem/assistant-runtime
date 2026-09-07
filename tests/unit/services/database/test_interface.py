"""Tests for DatabaseService lifecycle, sessions, and health checks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import DatabaseError
from assistant_runtime.services.database.interface import DatabaseService


def _make_mock_engine(*, connect_ok: bool = True):
    """Create a mock engine with proper async context manager for begin()."""
    mock_engine = MagicMock()
    mock_conn = AsyncMock()

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


class TestDatabaseServiceInit:
    """Constructor and initial state."""

    def test_initial_state(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)
        assert service._started is False
        assert service._healthy is False
        assert service._engine is None
        assert service._session_factory is None

    def test_stores_config(self):
        config = DatabaseConfig(host="custom-host")
        service = DatabaseService(config=config)
        assert service._config.host == "custom-host"


class TestDatabaseServiceStart:
    """Start lifecycle — engine creation and connectivity test."""

    @pytest.fixture
    def config(self):
        return DatabaseConfig()

    @pytest.fixture
    def service(self, config):
        return DatabaseService(config=config)

    async def test_start_creates_engine(self, service):
        mock_engine, _ = _make_mock_engine()
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ) as mock_create:
            await service.start()

            mock_create.assert_called_once()
            assert service._started is True
            assert service._engine is mock_engine

    async def test_start_with_config_values(self, config):
        service = DatabaseService(config=config)
        mock_engine, _ = _make_mock_engine()
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ) as mock_create:
            await service.start()

            call_kwargs = mock_create.call_args
            assert call_kwargs[0][0] == config.async_url
            assert call_kwargs[1]["pool_pre_ping"] is True
            assert call_kwargs[1]["pool_size"] == config.pool_size
            assert call_kwargs[1]["max_overflow"] == config.pool_overflow
            assert call_kwargs[1]["echo"] == config.echo

    async def test_start_creates_session_factory(self, service):
        mock_engine, _ = _make_mock_engine()
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            assert service._session_factory is not None

    async def test_start_healthy_on_successful_connection(self, service):
        mock_engine, _ = _make_mock_engine(connect_ok=True)
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            assert service._healthy is True

    async def test_start_degrades_gracefully_on_connection_failure(self, service):
        mock_engine, _ = _make_mock_engine(connect_ok=False)
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            assert service._started is True
            assert service._healthy is False


class TestDatabaseServiceStop:
    """Stop lifecycle — engine disposal."""

    async def test_stop_disposes_engine(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_engine, _ = _make_mock_engine()
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            await service.stop()

            mock_engine.dispose.assert_awaited_once()
            assert service._engine is None
            assert service._session_factory is None
            assert service._started is False
            assert service._healthy is False

    async def test_stop_when_not_started(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)
        await service.stop()
        assert service._started is False


class TestDatabaseServiceHealthCheck:
    """Health check reporting."""

    async def test_health_check_before_start(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)
        health = await service.health_check()
        assert health["healthy"] is False

    async def test_health_check_healthy(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_engine, _ = _make_mock_engine(connect_ok=True)
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            health = await service.health_check()

            assert health["healthy"] is True
            assert health["host"] == "localhost"

    async def test_health_check_detects_connection_loss(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_engine, _ = _make_mock_engine(connect_ok=True)
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
            assert service._healthy is True

            # Now make the connection fail on next health check
            mock_fail_cm = MagicMock()
            mock_fail_cm.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("gone"))
            mock_fail_cm.__aexit__ = AsyncMock(return_value=False)
            mock_engine.begin.return_value = mock_fail_cm

            health = await service.health_check()
            # Degraded, not unhealthy: every request path works without Postgres.
            assert health["healthy"] is True
            assert health["reachable"] is False


class TestDatabaseServiceSessionContext:
    """session_context() async context manager."""

    async def test_session_context_raises_when_not_started(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)
        with pytest.raises(DatabaseError, match="not started"):
            async with service.session_context():
                pass

    async def test_session_context_yields_session(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        service._session_factory = mock_session_factory
        service._started = True
        service._healthy = True

        async with service.session_context() as session:
            assert session is mock_session

    async def test_session_context_commits_on_success(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        service._session_factory = mock_session_factory
        service._started = True
        service._healthy = True

        async with service.session_context():
            pass

        mock_session.commit.assert_awaited_once()

    async def test_session_context_rolls_back_on_error(self):
        config = DatabaseConfig()
        service = DatabaseService(config=config)

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)

        service._session_factory = mock_session_factory
        service._started = True
        service._healthy = True

        with pytest.raises(ValueError, match="test error"):
            async with service.session_context():
                raise ValueError("test error")

        mock_session.rollback.assert_awaited_once()
        mock_session.commit.assert_not_awaited()
