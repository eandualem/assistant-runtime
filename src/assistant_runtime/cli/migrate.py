"""``assistant-runtime migrate``: create or update the Postgres schema.

The migrations ship inside the wheel, so an installed runtime can build its
schema without a checkout. The work is
:func:`assistant_runtime.services.database.migrations.migrate`, which a host
can also call from Python with settings it built itself; this command
supplies the ``DATABASE__*`` variables and ``.env``.
"""

from __future__ import annotations

import argparse
import asyncio


def cmd_migrate(args: argparse.Namespace) -> int:
    """Upgrade the database to ``--revision`` (``head`` by default)."""
    from assistant_runtime.config import AppSettings
    from assistant_runtime.services.database.exceptions import MigrationError
    from assistant_runtime.services.database.migrations import migrate

    # AppSettings is the only place DATABASE__* and .env are resolved;
    # DatabaseConfig on its own would silently return the defaults.
    database = AppSettings().database
    print(f"migrate: {database.host}:{database.port}/{database.name} -> {args.revision}")
    try:
        asyncio.run(migrate(database, args.revision))
    except MigrationError as exc:
        print(f"migrate: failed: {exc}")
        return 1
    print("migrate: done")
    return 0
