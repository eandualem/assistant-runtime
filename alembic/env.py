"""Alembic environment — async migration runner."""

import asyncio
from logging.config import fileConfig

from alembic import context
from dotenv import find_dotenv, load_dotenv
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine

from assistant_runtime.services.database.base import Base
from assistant_runtime.config import AppSettings
import assistant_runtime.services.database.models  # noqa: F401 — registers ORM models with Base.metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# A caller that built its own settings passes the URL (``migrations.migrate``);
# otherwise read DATABASE__* and .env. AppSettings, not DatabaseConfig
# directly: only the settings model reads them.
url = config.attributes.get("connection_url")
if url is None:
    load_dotenv(find_dotenv(usecwd=True))
    url = AppSettings().database.url()

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        # One upgrade at a time across processes (several workers migrating
        # on start): a second one waits here, then finds nothing left to do.
        connection.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended('assistant_runtime:migrate', 0))")
        )
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = create_async_engine(url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
