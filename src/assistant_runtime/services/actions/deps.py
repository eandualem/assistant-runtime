"""Dependency injection for the actions module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.services.actions.interface import ActionService


def get_action_service(request: Request) -> ActionService:
    """Access ActionService from app.state (created during lifespan)."""
    service = getattr(request.app.state, "action_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Action service not available")
    return service


ActionServiceDep = Annotated[ActionService, Depends(get_action_service)]
