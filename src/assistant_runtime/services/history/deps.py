"""Dependency injection for the history service module."""

from typing import Annotated

from fastapi import Depends, Request

from assistant_runtime.services.history.interface import HistoryService


def get_history_service(request: Request) -> HistoryService:
    """Access HistoryService from app.state (created during lifespan)."""
    return request.app.state.history_service


HistoryServiceDep = Annotated[HistoryService, Depends(get_history_service)]
