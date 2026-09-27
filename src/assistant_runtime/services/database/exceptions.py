"""Exception hierarchy for the database service module.

``DatabaseUnavailableError`` and ``MigrationError`` carry a machine-readable
``cause`` so a host can word them for its own users without parsing driver
text: ``unreachable``, ``auth_refused``, ``database_missing`` or
``migration_failed``.
"""

from typing import Literal

from assistant_runtime.base.exceptions import AssistantRuntimeError

DatabaseCause = Literal["unreachable", "auth_refused", "database_missing", "migration_failed"]

# Postgres error codes: the database does not exist; authentication failed;
# the role may not connect to the database.
_SQLSTATE_CAUSES: dict[str, DatabaseCause] = {
    "3D000": "database_missing",
    "28P01": "auth_refused",
    "28000": "auth_refused",
    "42501": "auth_refused",
}


class DatabaseError(AssistantRuntimeError):
    """Base exception for all database module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "database")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class DatabaseUnavailableError(DatabaseError):
    """Postgres cannot be used and ``DATABASE__REQUIRED`` forbids running without it."""

    def __init__(self, message: str, *, cause: DatabaseCause = "unreachable", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.cause: DatabaseCause = cause


class MigrationError(DatabaseError):
    """The schema could not be upgraded."""

    cause: DatabaseCause = "migration_failed"


def sqlstate(exc: BaseException) -> str | None:
    """The Postgres error code in an exception or its causes, if any."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for candidate in (current, getattr(current, "orig", None)):
            code = getattr(candidate, "sqlstate", None)
            if isinstance(code, str):
                return code
        current = current.__cause__ or current.__context__
    return None


def failure_cause(exc: BaseException) -> DatabaseCause:
    """Why a connection failed: a known Postgres refusal, else the server is unreachable."""
    return _SQLSTATE_CAUSES.get(sqlstate(exc) or "", "unreachable")
