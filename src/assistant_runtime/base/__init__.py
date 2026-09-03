"""Base module — foundation layer providing lifecycle, instrumentation, exceptions, and resilience."""

from assistant_runtime.base._logging import (
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
from assistant_runtime.base.exceptions import (
    AssistantRuntimeError,
    ConfigurationError,
    ExternalServiceError,
)
from assistant_runtime.base.instrument import instrument
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.base.protocols import LifecycleAware
from assistant_runtime.base.resilience import CircuitBreaker, retry_with_backoff

__all__ = [
    "CircuitBreaker",
    "ConfigurationError",
    "ExternalServiceError",
    "LifecycleAware",
    "LifecycleManager",
    "AssistantRuntimeError",
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
