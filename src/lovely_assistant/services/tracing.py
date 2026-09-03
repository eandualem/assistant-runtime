"""Optional Langfuse tracing integration.

Provides request-level traces and custom spans for non-LLM operations.
LLM calls are automatically traced by pydantic-ai's ``Agent.instrument_all()``.

All public functions are safe to call regardless of whether langfuse is installed
or configured. When tracing is disabled, context managers yield lightweight no-op
objects with zero overhead.

Configuration is via environment variables (standard langfuse convention):
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_tracing_enabled: bool = False
_langfuse_client: Any = None  # langfuse.Langfuse when available


# ---------------------------------------------------------------------------
# No-op handles
# ---------------------------------------------------------------------------


class _NoOpHandle:
    """No-op handle returned when tracing is disabled.

    Used for both traces and spans since both support the same interface.
    """

    def update_output(self, output: Any) -> None:  # noqa: ARG002
        pass

    def update(self, **kwargs: Any) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Live handles (wrap langfuse observations)
# ---------------------------------------------------------------------------


class _TraceHandle:
    """Wraps a langfuse root observation (trace) with a safe update interface."""

    def __init__(self, observation: Any) -> None:
        self._obs = observation

    def update_output(self, output: Any) -> None:
        try:
            self._obs.update(output=output)
        except Exception as e:
            logger.warning(f"[TRACING] Failed to update trace output: {e}")

    def update(self, **kwargs: Any) -> None:
        try:
            self._obs.update(**kwargs)
        except Exception as e:
            logger.warning(f"[TRACING] Failed to update trace: {e}")


class _SpanHandle:
    """Wraps a langfuse span observation with a safe update interface."""

    def __init__(self, observation: Any) -> None:
        self._obs = observation

    def update_output(self, output: Any) -> None:
        try:
            self._obs.update(output=output)
        except Exception as e:
            logger.warning(f"[TRACING] Failed to update span output: {e}")

    def update(self, **kwargs: Any) -> None:
        try:
            self._obs.update(**kwargs)
        except Exception as e:
            logger.warning(f"[TRACING] Failed to update span: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def initialize_tracing() -> bool:
    """Initialize Langfuse tracing and pydantic-ai instrumentation.

    Safe to call multiple times (idempotent). Returns ``True`` if tracing
    was successfully enabled, ``False`` otherwise (missing dependency,
    missing keys, auth failure, etc.).
    """
    global _tracing_enabled, _langfuse_client  # noqa: PLW0603

    if _tracing_enabled:
        logger.debug("[TRACING] Already initialized — skipping")
        return True

    logger.info("[TRACING] Initializing Langfuse tracing...")

    # Log environment variable status (presence only, never values)
    has_secret = bool(os.environ.get("LANGFUSE_SECRET_KEY"))
    has_public = bool(os.environ.get("LANGFUSE_PUBLIC_KEY"))
    host = os.environ.get("LANGFUSE_HOST", "(not set, will use default)")
    logger.info(
        f"[TRACING] Environment: LANGFUSE_SECRET_KEY={'set' if has_secret else 'MISSING'}, "
        f"LANGFUSE_PUBLIC_KEY={'set' if has_public else 'MISSING'}, "
        f"LANGFUSE_HOST={host}"
    )

    if not has_secret or not has_public:
        logger.warning("[TRACING] Missing required environment variables — tracing disabled")
        return False

    try:
        from langfuse import get_client
        from pydantic_ai import Agent

        client = get_client()
        logger.info("[TRACING] Langfuse client created, running auth check...")

        if not client.auth_check():
            logger.warning("[TRACING] Langfuse auth check failed — tracing disabled")
            return False

        Agent.instrument_all()

        _langfuse_client = client
        _tracing_enabled = True
        logger.info(f"[TRACING] Langfuse tracing enabled (host={host})")
        return True

    except ImportError:
        logger.warning("[TRACING] langfuse package not installed — tracing disabled")
        return False
    except Exception as e:
        logger.warning(f"[TRACING] Failed to initialize: {e}")
        return False


def shutdown_tracing() -> None:
    """Flush pending traces and shut down the langfuse client.

    Safe to call when tracing was never initialized.
    """
    global _tracing_enabled, _langfuse_client  # noqa: PLW0603

    if not _tracing_enabled or _langfuse_client is None:
        logger.debug("[TRACING] Shutdown called but tracing was not active")
        return

    try:
        _langfuse_client.flush()
        logger.info("[TRACING] Langfuse tracing shut down, pending traces flushed")
    except Exception as e:
        logger.warning(f"[TRACING] Error during tracing shutdown: {e}")
    finally:
        _tracing_enabled = False
        _langfuse_client = None


def is_tracing_enabled() -> bool:
    """Return whether tracing is currently active."""
    return _tracing_enabled


@contextmanager
def create_request_trace(
    *,
    session_id: str,
    model: str | None = None,
    is_continuation: bool = False,
    input_message: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    set_current_observation: bool = True,
):
    """Create a root trace for a full request lifecycle.

    Yields a handle with ``update_output()`` / ``update()`` methods.
    When tracing is disabled, yields a ``_NoOpHandle``.

    ``set_current_observation=False`` is intended for async generator flows
    where the request lifecycle spans multiple ``yield`` points. In that mode
    we avoid OpenTelemetry context attach/detach and end the observation
    manually, which is safe across async generator shutdown.
    """
    if not _tracing_enabled or _langfuse_client is None:
        yield _NoOpHandle()
        return

    # Build tags
    trace_tags = list(tags) if tags else []
    if model:
        trace_tags.append(f"model:{model}")
    if is_continuation:
        trace_tags.append("continuation")

    # Build metadata
    trace_metadata = dict(metadata) if metadata else {}

    try:
        if set_current_observation:
            cm = _langfuse_client.start_as_current_observation(
                as_type="span",
                name="agent-request",
                input=input_message,
                metadata=trace_metadata,
            )
            obs = cm.__enter__()
        else:
            cm = None
            obs = _langfuse_client.start_observation(
                as_type="span",
                name="agent-request",
                input=input_message,
                metadata=trace_metadata,
            )
        obs.update_trace(
            session_id=str(session_id),
            user_id=os.environ.get("LANGFUSE_USER_ID", "operator"),
            tags=trace_tags,
            input=input_message,
            metadata=trace_metadata,
        )
    except Exception as e:
        logger.warning(f"[TRACING] Failed to start request trace: {e}")
        yield _NoOpHandle()
        return

    try:
        yield _TraceHandle(obs)
    finally:
        try:
            if cm is not None:
                cm.__exit__(None, None, None)
            else:
                obs.end()
        except Exception as e:
            logger.warning(f"[TRACING] Error closing request trace: {e}")


@contextmanager
def create_span(
    name: str,
    *,
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Create a child span under the current trace.

    Yields a handle with ``update_output()`` / ``update()`` methods.
    When tracing is disabled, yields a ``_NoOpHandle``.
    """
    if not _tracing_enabled or _langfuse_client is None:
        yield _NoOpHandle()
        return

    try:
        cm = _langfuse_client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input_data,
            metadata=metadata,
        )
        obs = cm.__enter__()
    except Exception as e:
        logger.warning(f"[TRACING] Failed to start span '{name}': {e}")
        yield _NoOpHandle()
        return

    try:
        yield _SpanHandle(obs)
    finally:
        try:
            cm.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"[TRACING] Error closing span '{name}': {e}")
