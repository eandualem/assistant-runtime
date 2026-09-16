"""Dependency injection for the heartbeat module."""

from fastapi import Request

from assistant_runtime.app.heartbeat.interface import HeartbeatService


def get_heartbeat_service(request: Request) -> HeartbeatService:
    """Access HeartbeatService from app.state."""
    return request.app.state.heartbeat_service
