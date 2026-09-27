"""Exception hierarchy for the actions module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class ActionError(AssistantRuntimeError):
    """Base exception for all action module errors."""

    error_code = "action_error"

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "actions")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class ActionNotFoundError(ActionError):
    """No action or confirmation with that id."""

    error_code = "action_not_found"


class ActionConflictError(ActionError):
    """The record is not in the state the change expects (status, revision, or a taken id)."""

    error_code = "action_conflict"
