"""Dependency injection for the streaming module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.app.streaming.interface import StreamingService


def get_streaming_service(request: Request) -> StreamingService:
    """Access StreamingService from app.state (created during lifespan)."""
    return request.app.state.streaming_service


StreamingServiceDep = Annotated[StreamingService, Depends(get_streaming_service)]
