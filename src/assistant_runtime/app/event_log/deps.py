"""Dependency injection for the event log module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.app.event_log.interface import EventLogService


def get_event_log_service(request: Request) -> EventLogService:
    """Access EventLogService from app.state (created during lifespan)."""
    service = getattr(request.app.state, "event_log_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Event log not available")
    return service


EventLogServiceDep = Annotated[EventLogService, Depends(get_event_log_service)]
