"""Exception hierarchy for the tasks module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class TaskError(AssistantRuntimeError):
    """Base exception for all task module errors."""

    error_code = "task_error"

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "tasks")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class TasksDisabledError(TaskError):
    """``TASKS__ENABLED`` is off."""

    error_code = "tasks_disabled"


class TaskNotFoundError(TaskError):
    """No task with that id is visible to the caller."""

    error_code = "task_not_found"


class TaskLimitError(TaskError):
    """Too many tasks are queued or running already."""

    error_code = "task_limit"
