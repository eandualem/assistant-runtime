"""Instrumentation — @instrument decorator for async function timing and tracking."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from loguru import logger


def instrument(
    operation: str | None = None,
    module: str | None = None,
) -> Callable:
    """Decorator that instruments async functions with timing and success/failure tracking.

    Emits structured log events with operation name, module, duration_ms, and success status.

    Args:
        operation: Override the operation name (defaults to function name).
        module: Override the module name (defaults to first segment of __module__).
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__
        mod_name = module or func.__module__.split(".")[-2] if "." in func.__module__ else "root"

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "operation.success",
                    module=mod_name,
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    success=True,
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.warning(
                    "operation.failure",
                    module=mod_name,
                    operation=op_name,
                    duration_ms=round(duration_ms, 2),
                    success=False,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator
