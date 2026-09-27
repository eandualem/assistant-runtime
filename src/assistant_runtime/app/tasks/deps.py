"""Dependency injection for the tasks module."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request

from assistant_runtime.app.tasks.interface import TaskService


def get_task_service(request: Request) -> TaskService:
    """Access TaskService from app.state (created during lifespan)."""
    service = getattr(request.app.state, "task_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Task service not available")
    return service


TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]
