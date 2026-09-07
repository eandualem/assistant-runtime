"""Dependency injection for the database service module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.services.database.interface import DatabaseService


def get_database_service(request: Request) -> DatabaseService:
    """Access DatabaseService from app.state (created during lifespan).

    Raises 503 when the database module was not registered or Postgres is unreachable.
    """
    db: DatabaseService | None = getattr(request.app.state, "database_service", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    if not getattr(db, "healthy", True):
        raise HTTPException(status_code=503, detail="Database not reachable")
    return db


DatabaseServiceDep = Annotated[DatabaseService, Depends(get_database_service)]
