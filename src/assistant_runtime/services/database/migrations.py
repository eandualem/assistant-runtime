"""Schema migrations and status, callable from Python.

The migrations ship inside the wheel. :func:`migrate` upgrades the database
described by a :class:`DatabaseConfig` that the caller built, so a host that
composes its settings in code does not depend on ``DATABASE__*`` variables;
the ``assistant-runtime migrate`` command and ``DATABASE__MIGRATE_ON_START``
use the same function.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import (
    DatabaseCause,
    DatabaseUnavailableError,
    MigrationError,
    failure_cause,
    sqlstate,
)


def migrations_dir() -> Path | None:
    """The Alembic script directory: the packaged copy, else a source checkout."""
    packaged = Path(__file__).resolve().parents[2] / "_migrations"
    if (packaged / "env.py").is_file():
        return packaged
    checkout = Path.cwd() / "alembic"
    if (checkout / "env.py").is_file():
        return checkout
    return None


# Alembic's ``context`` and ``op`` are process-global: two upgrades running
# at once would read each other's connection.
_ALEMBIC_LOCK = threading.Lock()


def _upgrade(config, revision: str) -> None:
    from alembic import command

    with _ALEMBIC_LOCK:
        command.upgrade(config, revision)


def _alembic_config(directory: Path, database: DatabaseConfig):
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", str(directory))
    # An attribute, not a main option: the ini parser would read the
    # escaped password's '%' as interpolation.
    config.attributes["connection_url"] = database.url()
    return config


async def migrate(database: DatabaseConfig, revision: str = "head") -> None:
    """Upgrade ``database`` to ``revision``; raise :class:`MigrationError` on failure."""
    directory = migrations_dir()
    if directory is None:
        raise MigrationError("The packaged schema migrations are missing from this install")
    config = _alembic_config(directory, database)
    try:
        # Alembic's environment runs its own event loop.
        await asyncio.to_thread(_upgrade, config, revision)
    except Exception as exc:
        raise MigrationError(f"Schema upgrade to {revision} failed: {exc}") from exc


async def create_database(database: DatabaseConfig) -> bool:
    """Create the configured database; False when it already exists.

    Connects to the server's ``postgres`` maintenance database with the same
    role. A role that may not create databases raises
    :class:`DatabaseUnavailableError` with ``cause="database_missing"``.
    """
    maintenance = database.model_copy(update={"name": "postgres"})
    engine = create_async_engine(maintenance.url(), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            name = conn.dialect.identifier_preparer.quote(database.name)
            # Raw SQL: a quoted name may contain ':', which text() reads as a parameter.
            await conn.exec_driver_sql(f"CREATE DATABASE {name}")
        return True
    except Exception as exc:
        # Another process may have created it meanwhile; Postgres reports that
        # race as 42P04 or as a unique violation on its catalog.
        if sqlstate(exc) == "42P04" or await _exists(engine, database.name):
            return False
        # Whatever stopped it (no CREATEDB right, ...), the database is still missing.
        raise DatabaseUnavailableError(
            f"Database {database.name!r} is missing and could not be created: {exc}",
            cause="database_missing",
        ) from exc
    finally:
        await engine.dispose()


async def _exists(engine: Any, name: str) -> bool:
    try:
        async with engine.connect() as conn:
            found = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": name}
            )
    except Exception:
        return False
    return found is not None


@dataclass(frozen=True)
class SchemaStatus:
    """Whether the database answers, and its revision against the packaged head.

    ``state`` names the case: ``unreachable`` (no server), ``auth_refused``,
    ``database_missing`` (created at start in required mode with
    ``migrate_on_start``), ``behind`` (upgraded at start with
    ``migrate_on_start``) or ``ready``. ``error`` keeps the underlying cause.
    """

    reachable: bool
    """The server answered (even if it refused the role or lacks the database)."""
    current: str | None = None
    head: str | None = None
    error: str | None = None
    cause: DatabaseCause | None = None

    @property
    def state(self) -> str:
        if self.cause is not None:
            return self.cause
        if not self.reachable:
            return "unreachable"
        return "ready" if self.head is not None and self.current == self.head else "behind"

    @property
    def up_to_date(self) -> bool:
        return self.state == "ready"


def _head_revision() -> str | None:
    directory = migrations_dir()
    if directory is None:
        return None
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(directory)).get_current_head()


async def schema_status(database: DatabaseConfig, *, timeout: float = 5.0) -> SchemaStatus:
    """Probe the database and read its Alembic revision. Never raises."""
    try:
        head = _head_revision()
    except Exception:
        head = None  # the packaged scripts could not be read; the probe still says why not
    engine = None
    try:
        engine = create_async_engine(database.url())
        async with asyncio.timeout(timeout), engine.connect() as conn:
            has_table = await conn.scalar(text("SELECT to_regclass('alembic_version') IS NOT NULL"))
            current = (
                await conn.scalar(text("SELECT version_num FROM alembic_version"))
                if has_table
                else None
            )
    except Exception as exc:
        cause = failure_cause(exc)
        return SchemaStatus(
            reachable=cause != "unreachable",
            head=head,
            error=str(exc) or type(exc).__name__,
            cause=cause,
        )
    finally:
        if engine is not None:
            await engine.dispose()
    return SchemaStatus(reachable=True, current=current, head=head)
