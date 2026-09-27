"""Exception hierarchy for the event log module."""

from assistant_runtime.base.exceptions import AssistantRuntimeError


class EventLogError(AssistantRuntimeError):
    """Base exception for all event log errors."""

    error_code = "event_log_error"

    def __init__(self, message: str, **kwargs) -> None:
        kwargs.setdefault("category", "event_log")
        kwargs.setdefault("severity", "medium")
        kwargs.setdefault("retry_allowed", False)
        super().__init__(message, **kwargs)


class EventNotFoundError(EventLogError):
    """No event with that id."""

    error_code = "event_not_found"


class EventConflictError(EventLogError):
    """The change does not fit the event (its direction, or a status it has passed)."""

    error_code = "event_conflict"
