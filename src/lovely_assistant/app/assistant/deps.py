"""Dependency injection for the assistant module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.app.assistant.interface import AssistantService


def get_assistant_service(request: Request) -> AssistantService:
    """Access AssistantService from app.state (created during lifespan)."""
    return request.app.state.assistant_service


AssistantServiceDep = Annotated[AssistantService, Depends(get_assistant_service)]
