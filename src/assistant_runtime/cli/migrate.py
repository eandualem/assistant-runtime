"""``assistant-runtime migrate``: create or update the Postgres schema.

The migrations ship inside the wheel, so an installed runtime can build its
schema without a checkout. ``alembic/env.py`` reads the connection settings
from ``DatabaseConfig`` (the ``DATABASE__*`` variables and ``.env``), so only
the script location has to be supplied here.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def migrations_dir() -> Path | None:
    """The Alembic script directory: the packaged copy, else a source checkout."""
    packaged = Path(__file__).resolve().parent.parent / "_migrations"
    if (packaged / "env.py").is_file():
        return packaged
    checkout = Path.cwd() / "alembic"
    if (checkout / "env.py").is_file():
        return checkout
    return None


def cmd_migrate(args: argparse.Namespace) -> int:
    """Upgrade the database to ``--revision`` (``head`` by default)."""
    directory = migrations_dir()
    if directory is None:
        print(
            "migrate: no migrations found. Run this from a checkout, or reinstall "
            "assistant-runtime from a wheel that ships them.",
        )
        return 1

    from alembic import command
    from alembic.config import Config

    from assistant_runtime.services.database.config import DatabaseConfig

    database = DatabaseConfig()
    config = Config()
    config.set_main_option("script_location", str(directory))
    print(f"migrate: {database.host}:{database.port}/{database.name} -> {args.revision}")
    try:
        command.upgrade(config, args.revision)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not raise
        print(f"migrate: failed: {exc}")
        return 1
    print("migrate: done")
    return 0
