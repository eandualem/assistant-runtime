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

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import MigrationError


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
        raise MigrationError(
            "No migrations found: reinstall assistant-runtime from a wheel that ships them, "
            "or run from a checkout."
        )
    config = _alembic_config(directory, database)
    try:
        # Alembic's environment runs its own event loop.
        await asyncio.to_thread(_upgrade, config, revision)
    except Exception as exc:
        raise MigrationError(f"Schema upgrade to {revision} failed: {exc}") from exc


@dataclass(frozen=True)
class SchemaStatus:
    """Whether the database answers, and its revision against the packaged head."""

    reachable: bool
    current: str | None = None
    head: str | None = None
    error: str | None = None

    @property
    def up_to_date(self) -> bool:
        return self.reachable and self.head is not None and self.current == self.head


def _head_revision() -> str | None:
    directory = migrations_dir()
    if directory is None:
        return None
    from alembic.script import ScriptDirectory

    return ScriptDirectory(str(directory)).get_current_head()


async def schema_status(database: DatabaseConfig, *, timeout: float = 5.0) -> SchemaStatus:
    """Probe the database and read its Alembic revision. Never raises."""
    head = _head_revision()
    engine = create_async_engine(database.url())
    try:
        async with asyncio.timeout(timeout), engine.connect() as conn:
            has_table = await conn.scalar(text("SELECT to_regclass('alembic_version') IS NOT NULL"))
            current = (
                await conn.scalar(text("SELECT version_num FROM alembic_version"))
                if has_table
                else None
            )
    except Exception as exc:
        return SchemaStatus(reachable=False, head=head, error=str(exc) or type(exc).__name__)
    finally:
        await engine.dispose()
    return SchemaStatus(reachable=True, current=current, head=head)
