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
        with pytest.raises(MigrationError, match="No migrations found"):
            await migrate(DatabaseConfig())


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
