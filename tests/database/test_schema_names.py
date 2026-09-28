"""The migrated schema names every constraint and index as the models do.

Opt-in: runs only when ``TEST_DATABASE_URL`` names a Postgres server whose
role may create databases (CI's database job sets it), and is skipped
otherwise, so the suite still needs no services. It migrates one scratch
database to head, builds another from the models with ``create_all``, and
compares the names in both.
"""

from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine

import assistant_runtime.services.database.models  # noqa: F401 - registers the tables
from assistant_runtime.services.database.base import Base
from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.migrations import create_database, migrate

URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set")

_NAMES = text(
    """
    select t.relname, c.conname, c.contype::text
    from pg_constraint c
    join pg_class t on t.oid = c.conrelid
    join pg_namespace n on n.oid = t.relnamespace
    where n.nspname = current_schema()
    union all
    select tablename, indexname, 'index'
    from pg_indexes
    where schemaname = current_schema()
    """
)


def _config(name: str) -> DatabaseConfig:
    url = make_url(URL)
    return DatabaseConfig(
        host=url.host or "localhost",
        port=url.port or 5432,
        user=url.username or "postgres",
        password=url.password or "",
        name=name,
    )


async def _names(database: DatabaseConfig) -> set[tuple[str, str, str]]:
    engine = create_async_engine(database.url())
    try:
        async with engine.connect() as conn:
            rows = (await conn.execute(_NAMES)).all()
    finally:
        await engine.dispose()
    return {tuple(row) for row in rows if row[0] != "alembic_version"}


async def _drop(*names: str) -> None:
    engine = create_async_engine(_config("postgres").url(), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            for name in names:
                await conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    finally:
        await engine.dispose()


async def test_migrations_name_everything_as_the_models_do():
    suffix = uuid.uuid4().hex[:8]
    migrated, modelled = _config(f"schema_migrated_{suffix}"), _config(f"schema_models_{suffix}")
    try:
        await create_database(migrated)
        await create_database(modelled)
        await migrate(migrated)
        engine = create_async_engine(modelled.url())
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()
        from_migrations, from_models = await _names(migrated), await _names(modelled)
        assert sorted(from_migrations - from_models) == sorted(from_models - from_migrations) == []
    finally:
        await _drop(migrated.name, modelled.name)
