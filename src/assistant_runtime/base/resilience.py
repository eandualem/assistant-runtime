"""Resilience patterns — retry with backoff and circuit breaker."""

from collections.abc import Callable

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
