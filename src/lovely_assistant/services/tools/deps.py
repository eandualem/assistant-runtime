"""Dependency injection for the tool service module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.services.tools.interface import ToolService


def get_tool_service(request: Request) -> ToolService:
    """Access ToolService from app.state (created during lifespan)."""
    return request.app.state.tool_service


ToolServiceDep = Annotated[ToolService, Depends(get_tool_service)]
