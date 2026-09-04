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
from assistant_runtime.services.database.exceptions import DatabaseError


class DatabaseService:
    """Async database engine and session management. Implements LifecycleAware."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._started = False
        self._healthy = False

    async def start(self) -> None:
        """Create engine, session factory, and test connectivity."""
        self._engine = create_async_engine(
            self._config.async_url,
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
            logger.warning(
                "Database not reachable — degraded mode",
                host=self._config.host,
                port=self._config.port,
                error=str(e),
            )
            self._healthy = False

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
        """Report health status with live connectivity check."""
        if not self._started or self._engine is None:
            return {"healthy": False}
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            self._healthy = True
        except Exception:
            self._healthy = False
        return {"healthy": self._healthy, "host": self._config.host}

    @property
    def healthy(self) -> bool:
        """Whether the database is reachable."""
        return self._healthy

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
