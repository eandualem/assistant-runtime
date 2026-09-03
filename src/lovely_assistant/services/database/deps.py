"""Dependency injection for the database service module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from lovely_assistant.services.database.interface import DatabaseService


def get_database_service(request: Request) -> DatabaseService:
    """Access DatabaseService from app.state (created during lifespan).

    Raises 503 when the database module was not registered.
    """
    db: DatabaseService | None = getattr(request.app.state, "database_service", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    return db


DatabaseServiceDep = Annotated[DatabaseService, Depends(get_database_service)]
