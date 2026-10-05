"""Task endpoints — start, list, read and cancel background tasks.

A task runs a turn in its own session outside any conversation turn (see
``app/tasks``). Callers see their own tasks; an administrator sees all.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from assistant_runtime.app.access.deps import PrincipalDep
from assistant_runtime.app.tasks.deps import TaskServiceDep
from assistant_runtime.app.tasks.exceptions import (
    TaskError,
    TaskLimitError,
    TaskNotFoundError,
    TasksDisabledError,
)
from assistant_runtime.services.artifacts.exceptions import UnknownProfileError

router = APIRouter(prefix="/tasks", tags=["tasks"])

_STATUS = {
    TasksDisabledError: 503,
    TaskNotFoundError: 404,
    TaskLimitError: 429,
    UnknownProfileError: 404,
}


class StartTaskRequest(BaseModel):
    task: str = Field(..., min_length=1)
    profile: str | None = None
    subject: str | None = None
    context: str | None = None
    parent_session_id: str | None = Field(default=None, max_length=64)


def _http_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=_STATUS.get(type(exc), 422), detail=str(exc))


@router.post("", status_code=202)
async def start_task(
    body: StartTaskRequest, tasks: TaskServiceDep, principal: PrincipalDep
) -> dict:
    """Queue a task; it runs in the background and ends with ``task_finished``."""
    try:
        record = await tasks.start_task(
            body.task,
            profile=body.profile,
            subject=body.subject,
            context=body.context,
            parent_session_id=body.parent_session_id,
            principal=principal,
        )
    except (TaskError, UnknownProfileError) as e:
        raise _http_error(e) from e
    return record.to_dict()


@router.get("")
async def list_tasks(
    tasks: TaskServiceDep,
    principal: PrincipalDep,
    parent_session_id: str | None = None,
    status: str | None = Query(
        None, pattern="^(queued|running|done|failed|cancelled|interrupted)$"
    ),
    limit: int = Query(50, ge=1, le=500),
    before: str | None = None,
) -> list[dict]:
    """Newest first: the caller's tasks (every task for an administrator).

    ``before`` names a task: only older ones are listed (``404`` when it is unknown).
    """
    try:
        records = await tasks.list(
            principal,
            parent_session_id=parent_session_id,
            status=status,
            limit=limit,
            before=before,
        )
    except TaskError as e:
        raise _http_error(e) from e
    return [record.to_dict() for record in records]


@router.get("/{task_id}")
async def get_task(task_id: str, tasks: TaskServiceDep, principal: PrincipalDep) -> dict:
    try:
        return (await tasks.get(task_id, principal)).to_dict()
    except TaskError as e:
        raise _http_error(e) from e


@router.post("/{task_id}/cancel")
async def cancel_task(task_id: str, tasks: TaskServiceDep, principal: PrincipalDep) -> dict:
    """Stop a queued or running task; a finished one is returned unchanged."""
    try:
        return (await tasks.cancel(task_id, principal)).to_dict()
    except TaskError as e:
        raise _http_error(e) from e
