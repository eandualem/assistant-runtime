"""Exception hierarchy for the database service module."""

from lovely_assistant.base.exceptions import LovelyAssistantError


class DatabaseError(LovelyAssistantError):
    """Base exception for all database module errors."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "database")
        kwargs.setdefault("severity", "high")
        super().__init__(message, **kwargs)


class DatabaseConnectionError(DatabaseError):
    """Failed to connect to database."""

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("severity", "critical")
        super().__init__(message, **kwargs)
