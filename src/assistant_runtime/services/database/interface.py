"""DatabaseService — async engine and session management.

Public facade for the database module. Creates and manages the async engine,
provides session factories for FastAPI Depends() injection and context manager usage.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from loguru import logger
from sqlalchemy import text
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
)
from assistant_runtime.services.database.migrations import migrate


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
        :class:`DatabaseUnavailableError`. ``migrate_on_start`` upgrades the
        schema once the database answers.
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
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            self._healthy = True
            logger.info(
                "Database service started",
                host=self._config.host,
                port=self._config.port,
                database=self._config.name,
            )
        except Exception as e:
            self._healthy = False
            if self._config.required:
                await self.stop()
                raise DatabaseUnavailableError(
                    f"Postgres is not reachable at {self._config.host}:{self._config.port}/"
                    f"{self._config.name} and DATABASE__REQUIRED is set: {e}"
                ) from e
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
        """
        if not self._started or self._engine is None:
            return {"healthy": False, "reachable": False}
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            self._healthy = True
        except Exception:
            self._healthy = False
        healthy = self._healthy or not self._config.required
        return {"healthy": healthy, "reachable": self._healthy, "host": self._config.host}

    @property
    def healthy(self) -> bool:
        """Whether the database is reachable."""
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
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
