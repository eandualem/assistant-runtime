"""Base module — foundation layer providing lifecycle, instrumentation, exceptions, and resilience."""

from lovely_assistant.base._logging import (
    TAG_DB,
    TAG_HISTORY,
    TAG_LLM,
    TAG_REQUEST,
    TAG_SESSION,
    TAG_STREAM,
    TAG_TOOLS,
    truncate,
    truncate_payload,
)
from lovely_assistant.base.exceptions import (
    ConfigurationError,
    ExternalServiceError,
    LovelyAssistantError,
)
from lovely_assistant.base.instrument import instrument
from lovely_assistant.base.lifecycle import LifecycleManager
from lovely_assistant.base.protocols import LifecycleAware
from lovely_assistant.base.resilience import CircuitBreaker, retry_with_backoff

__all__ = [
    "CircuitBreaker",
    "ConfigurationError",
    "ExternalServiceError",
    "LifecycleAware",
    "LifecycleManager",
    "LovelyAssistantError",
    "TAG_DB",
    "TAG_HISTORY",
    "TAG_LLM",
    "TAG_REQUEST",
    "TAG_SESSION",
    "TAG_STREAM",
    "TAG_TOOLS",
    "instrument",
    "retry_with_backoff",
    "truncate",
    "truncate_payload",
]
