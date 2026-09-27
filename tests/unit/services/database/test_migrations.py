"""Tests for the public schema migration and status functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.services.database import migrations
from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import MigrationError
from assistant_runtime.services.database.migrations import SchemaStatus, migrate, schema_status


class TestMigrate:
    async def test_upgrades_the_given_database(self, monkeypatch, tmp_path):
        """The caller's config reaches alembic, not DATABASE__* from the environment."""
        import alembic.command

        monkeypatch.setenv("DATABASE__HOST", "from-env")
        monkeypatch.setattr(migrations, "migrations_dir", lambda: tmp_path)
        seen = {}
        monkeypatch.setattr(
            alembic.command,
            "upgrade",
            lambda config, revision: seen.update(
                revision=revision, url=config.attributes["connection_url"]
            ),
        )

        await migrate(DatabaseConfig(host="/var/run/postgresql", password=""), "0027")

        assert seen["revision"] == "0027"
        assert seen["url"].query == {"host": "/var/run/postgresql"}
        assert seen["url"].password is None

    async def test_failure_is_a_migration_error(self, monkeypatch, tmp_path):
        import alembic.command

        monkeypatch.setattr(migrations, "migrations_dir", lambda: tmp_path)

        def fail(config, revision):
            raise RuntimeError("relation already exists")

        monkeypatch.setattr(alembic.command, "upgrade", fail)
        with pytest.raises(MigrationError, match="relation already exists"):
            await migrate(DatabaseConfig())

    async def test_missing_migrations_is_a_migration_error(self, monkeypatch):
        monkeypatch.setattr(migrations, "migrations_dir", lambda: None)
        with pytest.raises(MigrationError, match="migrations are missing"):
            await migrate(DatabaseConfig())


async def test_concurrent_migrations_run_one_at_a_time(monkeypatch, tmp_path):
    """Alembic's context is process-global; overlapping upgrades would mix connections."""
    import threading
    import time

    import alembic.command

    monkeypatch.setattr(migrations, "migrations_dir", lambda: tmp_path)
    running, overlaps = [0], []
    guard = threading.Lock()

    def upgrade(config, revision):
        with guard:
            running[0] += 1
            overlaps.append(running[0])
        time.sleep(0.05)
        with guard:
            running[0] -= 1

    monkeypatch.setattr(alembic.command, "upgrade", upgrade)
    import asyncio

    await asyncio.gather(*(migrate(DatabaseConfig(name=f"db{i}")) for i in range(3)))
    assert overlaps == [1, 1, 1]


def _engine(*, connect_error: Exception | None = None, revision: str | None = "0027"):
    conn = MagicMock()
    conn.scalar = AsyncMock(side_effect=[revision is not None, revision])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn, side_effect=connect_error)
    cm.__aexit__ = AsyncMock(return_value=False)
    engine = MagicMock()
    engine.connect.return_value = cm
    engine.dispose = AsyncMock()
    return engine


class TestSchemaStatus:
    _ENGINE = "assistant_runtime.services.database.migrations.create_async_engine"

    async def test_current_against_head(self):
        engine = _engine(revision="0026")
        with (
            patch(self._ENGINE, return_value=engine),
            patch.object(migrations, "_head_revision", return_value="0027"),
        ):
            status = await schema_status(DatabaseConfig())
        assert status == SchemaStatus(reachable=True, current="0026", head="0027")
        assert status.up_to_date is False
        engine.dispose.assert_awaited_once()

    async def test_empty_database_has_no_revision(self):
        with (
            patch(self._ENGINE, return_value=_engine(revision=None)),
            patch.object(migrations, "_head_revision", return_value="0027"),
        ):
            status = await schema_status(DatabaseConfig())
        assert status.reachable is True
        assert status.current is None

    async def test_unreachable_never_raises(self):
        engine = _engine(connect_error=ConnectionRefusedError("refused"))
        with (
            patch(self._ENGINE, return_value=engine),
            patch.object(migrations, "_head_revision", return_value="0027"),
        ):
            status = await schema_status(DatabaseConfig())
        assert status.reachable is False
        assert "refused" in (status.error or "")
        assert status.up_to_date is False

    def test_packaged_head_is_found(self):
        assert migrations._head_revision() is not None


class _PgError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.sqlstate = code


class TestStates:
    _ENGINE = "assistant_runtime.services.database.migrations.create_async_engine"

    async def _status(self, error=None, revision="0027", head="0027"):
        engine = _engine(connect_error=error, revision=revision)
        with (
            patch(self._ENGINE, return_value=engine),
            patch.object(migrations, "_head_revision", return_value=head),
        ):
            return await schema_status(DatabaseConfig())

    async def test_each_case_is_named(self):
        assert (await self._status()).state == "ready"
        assert (await self._status(revision="0026")).state == "behind"
        missing = await self._status(_PgError("3D000"))
        assert (missing.state, missing.reachable, missing.cause) == (
            "database_missing",
            True,
            "database_missing",
        )
        assert (await self._status(_PgError("28P01"))).state == "auth_refused"
        unreachable = await self._status(ConnectionRefusedError("refused"))
        assert (unreachable.state, unreachable.reachable) == ("unreachable", False)
        assert "refused" in unreachable.error


class TestCreateDatabase:
    _ENGINE = "assistant_runtime.services.database.migrations.create_async_engine"

    def _engine(self, execute_error=None, exists=None):
        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=execute_error)
        conn.scalar = AsyncMock(return_value=exists)
        conn.dialect.identifier_preparer.quote = lambda name: f'"{name}"'
        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=conn)
        cm.__aexit__ = AsyncMock(return_value=False)
        engine = MagicMock()
        engine.connect.return_value = cm
        engine.dispose = AsyncMock()
        return engine, conn

    async def test_creates_through_the_maintenance_database(self):
        from assistant_runtime.services.database.migrations import create_database

        engine, conn = self._engine()
        with patch(self._ENGINE, return_value=engine) as make:
            assert await create_database(DatabaseConfig(name="fleet_db")) is True
        assert make.call_args[0][0].database == "postgres"
        assert make.call_args[1]["isolation_level"] == "AUTOCOMMIT"
        assert 'CREATE DATABASE "fleet_db"' in str(conn.execute.await_args[0][0])

    async def test_a_concurrent_creator_is_not_an_error(self):
        from assistant_runtime.services.database.migrations import create_database

        engine, _ = self._engine(_PgError("42P04"))
        with patch(self._ENGINE, return_value=engine):
            assert await create_database(DatabaseConfig()) is False

    async def test_a_creator_that_lost_the_race_finds_the_database_there(self):
        """Postgres reports a concurrent CREATE DATABASE as a catalog unique violation."""
        from assistant_runtime.services.database.migrations import create_database

        engine, _ = self._engine(_PgError("23505"), exists=1)
        with patch(self._ENGINE, return_value=engine):
            assert await create_database(DatabaseConfig()) is False

    async def test_a_role_without_the_right_reports_the_database_missing(self):
        from assistant_runtime.services.database.exceptions import DatabaseUnavailableError
        from assistant_runtime.services.database.migrations import create_database

        engine, _ = self._engine(_PgError("42501"))
        with (
            patch(self._ENGINE, return_value=engine),
            pytest.raises(DatabaseUnavailableError, match="could not be created") as caught,
        ):
            await create_database(DatabaseConfig())
        assert caught.value.cause == "database_missing"
