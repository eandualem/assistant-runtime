"""The model's task tools: start, list, read and cancel background tasks.

Registered when ``TASKS__ENABLED`` is on. The session that starts a task is
its parent; the tools list and read tasks of the caller's own principal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from assistant_runtime.app.tasks.exceptions import TaskError
from assistant_runtime.services.artifacts.exceptions import ArtifactError
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition
from assistant_runtime.services.tools.request_context import (
    get_current_assistant_session_id,
    get_current_principal,
)

if TYPE_CHECKING:
    from assistant_runtime.app.tasks.interface import TaskService
    from assistant_runtime.services.tools.interface import ToolService


def _failure(error: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "error": str(error),
        "error_code": getattr(error, "error_code", "task_error"),
    }


def _summary(record: Any) -> dict[str, Any]:
    return {
        "task_id": record.id,
        "status": record.status,
        "profile": record.profile,
        "subject": record.subject,
        "task": record.task,
    }


def register_task_tools(tools: ToolService, tasks: TaskService) -> None:
    async def start_task(
        task: str, profile: str = "", subject: str = "", context: str = ""
    ) -> dict[str, Any]:
        try:
            record = await tasks.start_task(
                task,
                profile=profile or None,
                subject=subject or None,
                context=context or None,
                parent_session_id=get_current_assistant_session_id(),
                principal=get_current_principal(),
            )
        except (TaskError, ArtifactError) as e:
            return _failure(e)
        return {**_summary(record), "success": True}

    async def list_tasks(status: str = "") -> dict[str, Any]:
        try:
            records = await tasks.list(
                get_current_principal(),
                parent_session_id=get_current_assistant_session_id(),
                status=status or None,
            )
        except TaskError as e:
            return _failure(e)
        return {"tasks": [_summary(r) for r in records], "success": True}

    async def get_task(task_id: str) -> dict[str, Any]:
        try:
            record = await tasks.get(task_id, get_current_principal())
        except TaskError as e:
            return _failure(e)
        return {**record.to_dict(), "success": True}

    async def cancel_task(task_id: str) -> dict[str, Any]:
        try:
            record = await tasks.cancel(task_id, get_current_principal())
        except TaskError as e:
            return _failure(e)
        return {**_summary(record), "success": True}

    tools.register_backend_tool(
        ToolDefinition(
            name="start_task",
            description=(
                "Start a bounded piece of work that runs in the background while you keep "
                "talking: it gets its own session and returns an id at once. Name a profile "
                "for the kind of agent that should do it, and a subject when the work is "
                "about one (tasks about the same subject run one at a time). Check it later "
                "with get_task or list_tasks."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The work, stated fully"},
                    "profile": {"type": "string", "description": "Registered profile to run it"},
                    "subject": {"type": "string", "description": "Whom the work is about"},
                    "context": {"type": "string", "description": "Anything the task needs"},
                },
                "required": ["task"],
            },
            category=ToolCategory.BACKEND,
        ),
        start_task,
    )
    tools.register_backend_tool(
        ToolDefinition(
            name="list_tasks",
            description="The tasks this conversation started, newest first, with their status.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["queued", "running", "done", "failed", "cancelled", "interrupted"],
                    }
                },
            },
            category=ToolCategory.BACKEND,
        ),
        list_tasks,
    )
    tools.register_backend_tool(
        ToolDefinition(
            name="get_task",
            description="One task's record, with its result once it is done.",
            parameters_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        get_task,
    )
    tools.register_backend_tool(
        ToolDefinition(
            name="cancel_task",
            description="Stop a queued or running task.",
            parameters_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            category=ToolCategory.BACKEND,
        ),
        cancel_task,
    )
