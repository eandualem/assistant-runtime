"""Resilience patterns — retry with backoff and circuit breaker."""

import asyncio
import time
from collections.abc import Callable
from enum import Enum
from functools import wraps
from typing import Any

from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


def _make_before_sleep(name: str | None = None) -> Callable[[RetryCallState], None]:
    """Create a before_sleep callback that logs retry attempts with structured fields."""

    def _log_retry(retry_state: RetryCallState) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "[RETRY] Retrying after failure",
            name=name or "unknown",
            attempt=retry_state.attempt_number,
            wait=round(retry_state.next_action.sleep, 1) if retry_state.next_action else 0,
            error_type=type(exc).__name__ if exc else "unknown",
            error=str(exc) if exc else "unknown",
        )

    return _log_retry


def retry_with_backoff(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_on: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
    name: str | None = None,
) -> Callable:
    """Thin wrapper around tenacity retry with sensible defaults.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        min_wait: Minimum wait time in seconds between retries.
        max_wait: Maximum wait time in seconds between retries.
        retry_on: Tuple of exception types that trigger a retry.
        name: Optional name for log context (identifies which operation is retrying).
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
        before_sleep=_make_before_sleep(name),
    )


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Async circuit breaker — closed -> open -> half-open -> closed.

    Implements LifecycleAware protocol for integration with LifecycleManager.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 3,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state, considering recovery timeout."""
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self._recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
        return self._state

    def protect(self) -> Callable:
        """Decorator that wraps an async function with circuit breaking."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                current_state = self.state

                if current_state == CircuitState.OPEN:
                    raise CircuitOpenError(f"Circuit '{self.name}' is open — call rejected")

                if current_state == CircuitState.HALF_OPEN:
                    if self._half_open_calls >= self._half_open_max:
                        raise CircuitOpenError(f"Circuit '{self.name}' half-open limit reached")
                    self._half_open_calls += 1

                try:
                    result = await func(*args, **kwargs)
                    self._on_success()
                    return result
                except Exception as e:
                    self._on_failure()
                    raise e

            return wrapper

        return decorator

    def _on_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info("Circuit closed after recovery", circuit=self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    def _on_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit opened",
                circuit=self.name,
                failures=self._failure_count,
            )

    async def start(self) -> None:
        """Lifecycle start — reset state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0

    async def stop(self) -> None:
        """Lifecycle stop — no-op."""
        await asyncio.sleep(0)  # yield to event loop

    async def health_check(self) -> dict:
        """Report circuit breaker health."""
        return {
            "healthy": self.state != CircuitState.OPEN,
            "state": self.state.value,
            "failures": self._failure_count,
        }


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open."""
