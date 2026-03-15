"""Database service module."""

from lovely_assistant.services.database.deps import DatabaseServiceDep, get_database_service
from lovely_assistant.services.database.interface import DatabaseService

__all__ = [
    "DatabaseService",
    "DatabaseServiceDep",
    "get_database_service",
]
