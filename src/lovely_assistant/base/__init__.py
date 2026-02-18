"""Base module — foundation layer providing lifecycle, instrumentation, exceptions, and resilience."""

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
    "instrument",
    "retry_with_backoff",
]
