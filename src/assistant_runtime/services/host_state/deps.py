"""Dependency injection for the host state module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.services.host_state.interface import HostStateService


def get_host_state_service(request: Request) -> HostStateService:
    """Access HostStateService from app.state (created during lifespan)."""
    service = getattr(request.app.state, "host_state_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Host state not available")
    return service


HostStateServiceDep = Annotated[HostStateService, Depends(get_host_state_service)]
