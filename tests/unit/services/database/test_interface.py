"""Tests for DatabaseService lifecycle, sessions, and health checks."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import (
    DatabaseError,
    DatabaseUnavailableError,
    MigrationError,
)
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
            assert call_kwargs[0][0] == config.url()
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

    @pytest.mark.parametrize(
        ("config", "level", "message"),
        [
            (DatabaseConfig(), "INFO", "No database configured; sessions are kept in memory"),
            (
                DatabaseConfig(host="db.example"),
                "WARNING",
                "Database not reachable — degraded mode",
            ),
        ],
    )
    async def test_unreachable_database_is_a_warning_only_when_configured(
        self, config, level, message
    ):
        import io

        from loguru import logger

        logs = io.StringIO()
        sink = logger.add(logs, format="{level} {message}", level="INFO")
        mock_engine, _ = _make_mock_engine(connect_ok=False)
        try:
            with patch(
                "assistant_runtime.services.database.interface.create_async_engine",
                return_value=mock_engine,
            ):
                await DatabaseService(config=config).start()
        finally:
            logger.remove(sink)
        assert logs.getvalue().strip() == f"{level} {message}"


class TestDatabaseServiceRequired:
    """DATABASE__REQUIRED and DATABASE__MIGRATE_ON_START."""

    _ENGINE = "assistant_runtime.services.database.interface.create_async_engine"
    _MIGRATE = "assistant_runtime.services.database.interface.migrate"

    async def test_unreachable_required_database_fails_startup(self):
        service = DatabaseService(config=DatabaseConfig(required=True))
        mock_engine, _ = _make_mock_engine(connect_ok=False)
        with (
            patch(self._ENGINE, return_value=mock_engine),
            pytest.raises(DatabaseUnavailableError, match="DATABASE__REQUIRED"),
        ):
            await service.start()
        mock_engine.dispose.assert_awaited_once()
        assert service._started is False
        assert service.healthy is False

    async def test_migrates_once_reachable(self):
        config = DatabaseConfig(migrate_on_start=True)
        service = DatabaseService(config=config)
        mock_engine, _ = _make_mock_engine()
        with (
            patch(self._ENGINE, return_value=mock_engine),
            patch(self._MIGRATE, new=AsyncMock()) as migrate,
        ):
            await service.start()
        migrate.assert_awaited_once_with(config)
        assert service.healthy is True

    async def test_failed_migration_fails_startup(self):
        service = DatabaseService(config=DatabaseConfig(migrate_on_start=True))
        mock_engine, _ = _make_mock_engine()
        with (
            patch(self._ENGINE, return_value=mock_engine),
            patch(self._MIGRATE, new=AsyncMock(side_effect=MigrationError("bad revision"))),
            pytest.raises(MigrationError, match="bad revision"),
        ):
            await service.start()
        mock_engine.dispose.assert_awaited_once()

    async def test_no_migration_when_unreachable_and_optional(self):
        service = DatabaseService(config=DatabaseConfig(migrate_on_start=True))
        mock_engine, _ = _make_mock_engine(connect_ok=False)
        with (
            patch(self._ENGINE, return_value=mock_engine),
            patch(self._MIGRATE, new=AsyncMock()) as migrate,
        ):
            await service.start()
        migrate.assert_not_awaited()
        assert service.healthy is False

    async def test_required_database_lost_later_is_unhealthy(self):
        service = DatabaseService(config=DatabaseConfig(required=True))
        mock_engine, _ = _make_mock_engine()
        with patch(self._ENGINE, return_value=mock_engine):
            await service.start()
        mock_engine.begin.return_value.__aenter__ = AsyncMock(
            side_effect=ConnectionRefusedError("gone")
        )
        assert await service.health_check() == {
            "healthy": False,
            "reachable": False,
            "host": "localhost",
        }


class _PgError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.sqlstate = code


def _engine_failing_then(first: Exception | None, *, then_ok: bool = True):
    """An engine whose first connection test fails with ``first``; later ones succeed."""
    mock_engine, _ = _make_mock_engine()
    calls = {"n": 0}
    ok_cm = mock_engine.begin.return_value

    def begin():
        calls["n"] += 1
        if calls["n"] == 1 and first is not None:
            failing = MagicMock()
            failing.__aenter__ = AsyncMock(side_effect=first)
            failing.__aexit__ = AsyncMock(return_value=False)
            return failing
        if not then_ok:
            failing = MagicMock()
            failing.__aenter__ = AsyncMock(side_effect=first)
            failing.__aexit__ = AsyncMock(return_value=False)
            return failing
        return ok_cm

    mock_engine.begin.side_effect = begin
    return mock_engine


class TestMissingDatabase:
    _ENGINE = "assistant_runtime.services.database.interface.create_async_engine"
    _CREATE = "assistant_runtime.services.database.interface.create_database"
    _MIGRATE = "assistant_runtime.services.database.interface.migrate"

    async def test_required_migrating_runtime_creates_its_database_then_migrates(self):
        config = DatabaseConfig(required=True, migrate_on_start=True)
        service = DatabaseService(config=config)
        engine = _engine_failing_then(_PgError("3D000"))
        with (
            patch(self._ENGINE, return_value=engine),
            patch(self._CREATE, new=AsyncMock(return_value=True)) as create,
            patch(self._MIGRATE, new=AsyncMock()) as migrate,
        ):
            await service.start()
        create.assert_awaited_once_with(config)
        migrate.assert_awaited_once_with(config)
        assert service.healthy is True

    async def test_a_missing_database_without_migrate_on_start_is_not_created(self):
        service = DatabaseService(config=DatabaseConfig(required=True))
        engine = _engine_failing_then(_PgError("3D000"))
        with (
            patch(self._ENGINE, return_value=engine),
            patch(self._CREATE, new=AsyncMock()) as create,
            pytest.raises(DatabaseUnavailableError) as caught,
        ):
            await service.start()
        create.assert_not_awaited()
        assert caught.value.cause == "database_missing"

    async def test_a_database_that_cannot_be_created_fails_startup(self):
        service = DatabaseService(config=DatabaseConfig(required=True, migrate_on_start=True))
        engine = _engine_failing_then(_PgError("3D000"))
        refused = DatabaseUnavailableError(
            "missing and could not be created", cause="database_missing"
        )
        with (
            patch(self._ENGINE, return_value=engine),
            patch(self._CREATE, new=AsyncMock(side_effect=refused)),
            pytest.raises(DatabaseUnavailableError, match="could not be created") as caught,
        ):
            await service.start()
        assert caught.value.cause == "database_missing"
        engine.dispose.assert_awaited_once()

    @pytest.mark.parametrize(
        ("error", "cause"),
        [(_PgError("28P01"), "auth_refused"), (ConnectionRefusedError("no"), "unreachable")],
    )
    async def test_other_failures_carry_their_cause(self, error, cause):
        service = DatabaseService(config=DatabaseConfig(required=True, migrate_on_start=True))
        engine = _engine_failing_then(error, then_ok=False)
        with (
            patch(self._ENGINE, return_value=engine),
            patch(self._CREATE, new=AsyncMock()) as create,
            pytest.raises(DatabaseUnavailableError) as caught,
        ):
            await service.start()
        create.assert_not_awaited()
        assert caught.value.cause == cause


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
            # The probe only reports: calls keep trying the database.
            assert service.healthy is True

    async def test_a_database_back_after_a_degraded_start_stays_unused(self):
        service = DatabaseService(config=DatabaseConfig())
        mock_engine, _ = _make_mock_engine(connect_ok=False)
        with patch(
            "assistant_runtime.services.database.interface.create_async_engine",
            return_value=mock_engine,
        ):
            await service.start()
        mock_engine.begin.return_value.__aenter__ = AsyncMock()

        assert (await service.health_check())["reachable"] is True
        # The stores chose memory at startup, so nothing may start using it now.
        assert service.healthy is False


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

    async def test_session_context_reports_a_lost_connection_as_unreachable(self):
        service = DatabaseService(config=DatabaseConfig())
        session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
        service._session_factory = mock_session_factory
        service._started = True
        service._healthy = True

        # No connection can be made.
        refused = ConnectionRefusedError("Connection refused")
        session.connection.side_effect = refused
        with pytest.raises(DatabaseError, match="not reachable") as raised:
            async with service.session_context():
                pass
        assert raised.value.__cause__ is refused

        # No pooled connection frees up in time.
        exhausted = PoolTimeoutError("QueuePool limit reached")
        session.connection.side_effect = exhausted
        with pytest.raises(DatabaseError, match="not reachable") as raised:
            async with service.session_context():
                pass
        assert raised.value.__cause__ is exhausted
        session.connection.side_effect = None

        # The connection in use is dropped.
        dropped = DBAPIError("SELECT 1", {}, Exception("terminated"), connection_invalidated=True)
        with pytest.raises(DatabaseError, match="not reachable") as raised:
            async with service.session_context():
                raise dropped
        assert raised.value.__cause__ is dropped

        # The body's own errors keep their type.
        for error in (
            OSError("No such file"),
            IntegrityError("INSERT", {}, Exception("duplicate key")),
        ):
            with pytest.raises(type(error)):
                async with service.session_context():
                    raise error
