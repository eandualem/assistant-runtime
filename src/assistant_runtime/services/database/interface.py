"""DatabaseService — async engine and session management.

Public facade for the database module. Creates and manages the async engine,
provides session factories for FastAPI Depends() injection and context manager usage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.exceptions import (
    DatabaseError,
    DatabaseUnavailableError,
    failure_cause,
)
from assistant_runtime.services.database.migrations import create_database, migrate


class DatabaseService:
    """Async database engine and session management. Implements LifecycleAware."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._started = False
        self._healthy = False

    async def start(self) -> None:
        """Create engine, session factory, and test connectivity.

        An unreachable database leaves the runtime in memory, unless
        ``required`` is set: then startup fails with
        :class:`DatabaseUnavailableError` (its ``cause`` says why).
        ``migrate_on_start`` upgrades the schema once the database answers;
        with ``required`` too, a reachable server that lacks the database
        gets it created first.
        """
        self._engine = create_async_engine(
            self._config.url(),
            pool_pre_ping=True,
            pool_size=self._config.pool_size,
            max_overflow=self._config.pool_overflow,
            echo=self._config.echo,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        self._started = True

        # Test connectivity — degrade gracefully if DB is unavailable
        failure = await self._probe()
        if failure is not None and self._creates_missing(failure):
            try:
                await create_database(self._config)
            except DatabaseUnavailableError:
                await self.stop()
                raise
            logger.info("Created the missing database", database=self._config.name)
            failure = await self._probe()
        if failure is None:
            self._healthy = True
            logger.info(
                "Database service started",
                host=self._config.host,
                port=self._config.port,
                database=self._config.name,
            )
        else:
            e = failure
            self._healthy = False
            if self._config.required:
                await self.stop()
                raise DatabaseUnavailableError(
                    f"Postgres at {self._config.host}:{self._config.port}/{self._config.name} "
                    f"cannot be used and DATABASE__REQUIRED is set: {e}",
                    cause=failure_cause(e),
                ) from e
            if not self._config.model_fields_set:
                # Nothing configured and no local default Postgres: the
                # documented in-memory setup, not a fault.
                logger.info("No database configured; sessions are kept in memory")
                return
            logger.warning(
                "Database not reachable — degraded mode",
                host=self._config.host,
                port=self._config.port,
                error=str(e),
            )
            return
        if self._config.migrate_on_start:
            try:
                await migrate(self._config)
            except Exception:
                await self.stop()
                raise
            logger.info("Database schema upgraded to head")

    async def _probe(self) -> Exception | None:
        assert self._engine is not None
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            return e
        return None

    def _creates_missing(self, failure: Exception) -> bool:
        """Only a required database that migrates on start is created by the runtime."""
        return (
            self._config.required
            and self._config.migrate_on_start
            and failure_cause(failure) == "database_missing"
        )

    async def stop(self) -> None:
        """Dispose engine and clean up."""
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        self._started = False
        self._healthy = False
        logger.info("Database service stopped")

    async def health_check(self) -> dict:
        """Report reachability with a live check.

        Postgres is optional: an unreachable database is reported as
        ``reachable: false`` but does not make the runtime unhealthy, because
        every request path works without it — unless ``required`` is set,
        when it does. A service that never started is a genuine failure.
        The check only reports: it never changes :attr:`healthy`.
        """
        if not self._started or self._engine is None:
            return {"healthy": False, "reachable": False}
        failure = await self._probe()
        reachable = failure is None
        healthy = reachable or not self._config.required
        return {"healthy": healthy, "reachable": reachable, "host": self._config.host}

    @property
    def healthy(self) -> bool:
        """Whether startup found the database, so this process uses it.

        Fixed at startup: after an outage each call tries the database again
        (the pool reconnects), and a failed attempt is a :class:`DatabaseError`.
        """
        return self._healthy

    @property
    def required(self) -> bool:
        """Whether the runtime may not fall back to memory (``DATABASE__REQUIRED``)."""
        return self._config.required

    @asynccontextmanager
    async def session_context(self) -> AsyncIterator[AsyncSession]:
        """Context manager for non-DI use (background tasks, tests)."""
        if self._session_factory is None:
            raise DatabaseError("Database service not started")
        if not self._healthy:
            raise DatabaseError("Database not reachable")
        async with self._session_factory() as session:
            # A database lost after startup: no connection can be made, or the
            # one in use is dropped. Connecting first keeps the body's own
            # errors (an OSError from a file, say) out of this.
            try:
                await session.connection()
            except (OSError, DBAPIError) as e:
                raise DatabaseError("Database not reachable") from e
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                if isinstance(e, DBAPIError) and e.connection_invalidated:
                    raise DatabaseError("Database not reachable") from e
                raise
