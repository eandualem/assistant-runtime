"""Exception hierarchy for the database service module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class DatabaseError(AssistantRuntimeError):
    """Base exception for all database module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "database")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class DatabaseUnavailableError(DatabaseError):
    """Postgres is unreachable and ``DATABASE__REQUIRED`` forbids running without it."""


class MigrationError(DatabaseError):
    """The schema could not be upgraded."""
