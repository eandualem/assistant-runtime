"""Dependency injection for the heartbeat module."""

from typing import Annotated

from fastapi import Depends, Request

from lovely_assistant.app.heartbeat.interface import HeartbeatService


def get_heartbeat_service(request: Request) -> HeartbeatService:
    """Access HeartbeatService from app.state."""
    return request.app.state.heartbeat_service


HeartbeatServiceDep = Annotated[HeartbeatService, Depends(get_heartbeat_service)]
