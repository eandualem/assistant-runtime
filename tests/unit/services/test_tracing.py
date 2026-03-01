"""Unit tests for the Langfuse tracing module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lovely_assistant.services import tracing
from lovely_assistant.services.tracing import (
    _NoOpHandle,
    _SpanHandle,
    _TraceHandle,
    create_request_trace,
    create_span,
    initialize_tracing,
    is_tracing_enabled,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset module-level state before each test."""
    tracing._tracing_enabled = False
    tracing._langfuse_client = None
    yield
    tracing._tracing_enabled = False
    tracing._langfuse_client = None


# ---------------------------------------------------------------------------
# initialize_tracing
# ---------------------------------------------------------------------------


class TestInitializeTracing:
    def test_missing_env_vars_returns_false(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        assert initialize_tracing() is False
        assert is_tracing_enabled() is False

    def test_missing_secret_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        assert initialize_tracing() is False

    def test_missing_public_key_returns_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)

        assert initialize_tracing() is False

    def test_import_error_returns_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")

        # Setting a module to None in sys.modules causes ImportError on import
        with patch.dict("sys.modules", {"langfuse": None}):
            assert initialize_tracing() is False
            assert is_tracing_enabled() is False

    def test_auth_failure_returns_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")

        mock_client = MagicMock()
        mock_client.auth_check.return_value = False

        with patch("langfuse.get_client", return_value=mock_client):
            assert initialize_tracing() is False
            assert is_tracing_enabled() is False

    def test_success(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")

        mock_client = MagicMock()
        mock_client.auth_check.return_value = True

        with (
            patch("langfuse.get_client", return_value=mock_client),
            patch("pydantic_ai.Agent.instrument_all") as mock_instrument,
        ):
            result = initialize_tracing()

        assert result is True
        assert is_tracing_enabled() is True
        mock_client.auth_check.assert_called_once()
        mock_instrument.assert_called_once()

    def test_idempotent_second_call(self):
        tracing._tracing_enabled = True
        tracing._langfuse_client = MagicMock()

        assert initialize_tracing() is True

    def test_generic_exception_returns_false(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")

        with patch("langfuse.get_client", side_effect=RuntimeError("connection refused")):
            assert initialize_tracing() is False
            assert is_tracing_enabled() is False


# ---------------------------------------------------------------------------
# shutdown_tracing
# ---------------------------------------------------------------------------


class TestShutdownTracing:
    def test_not_initialized_is_noop(self):
        shutdown_tracing()
        assert is_tracing_enabled() is False

    def test_flushes_client(self):
        mock_client = MagicMock()
        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        shutdown_tracing()

        mock_client.flush.assert_called_once()
        assert is_tracing_enabled() is False
        assert tracing._langfuse_client is None

    def test_flush_error_still_resets_state(self):
        mock_client = MagicMock()
        mock_client.flush.side_effect = RuntimeError("flush failed")
        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        shutdown_tracing()

        assert is_tracing_enabled() is False
        assert tracing._langfuse_client is None


# ---------------------------------------------------------------------------
# _NoOpHandle
# ---------------------------------------------------------------------------


class TestNoOpHandle:
    def test_update_output_is_noop(self):
        handle = _NoOpHandle()
        handle.update_output("test output")  # should not raise

    def test_update_is_noop(self):
        handle = _NoOpHandle()
        handle.update(key="value", another="thing")  # should not raise


# ---------------------------------------------------------------------------
# _TraceHandle
# ---------------------------------------------------------------------------


class TestTraceHandle:
    def test_update_output_delegates(self):
        obs = MagicMock()
        handle = _TraceHandle(obs)
        handle.update_output("some output")
        obs.update.assert_called_once_with(output="some output")

    def test_update_delegates(self):
        obs = MagicMock()
        handle = _TraceHandle(obs)
        handle.update(name="test", tags=["a"])
        obs.update.assert_called_once_with(name="test", tags=["a"])

    def test_update_output_swallows_exceptions(self):
        obs = MagicMock()
        obs.update.side_effect = RuntimeError("langfuse error")
        handle = _TraceHandle(obs)
        handle.update_output("test")  # should not raise

    def test_update_swallows_exceptions(self):
        obs = MagicMock()
        obs.update.side_effect = RuntimeError("langfuse error")
        handle = _TraceHandle(obs)
        handle.update(key="value")  # should not raise


# ---------------------------------------------------------------------------
# _SpanHandle
# ---------------------------------------------------------------------------


class TestSpanHandle:
    def test_update_output_delegates(self):
        obs = MagicMock()
        handle = _SpanHandle(obs)
        handle.update_output("span output")
        obs.update.assert_called_once_with(output="span output")

    def test_update_delegates(self):
        obs = MagicMock()
        handle = _SpanHandle(obs)
        handle.update(foo="bar")
        obs.update.assert_called_once_with(foo="bar")

    def test_update_output_swallows_exceptions(self):
        obs = MagicMock()
        obs.update.side_effect = RuntimeError("boom")
        handle = _SpanHandle(obs)
        handle.update_output("test")  # should not raise

    def test_update_swallows_exceptions(self):
        obs = MagicMock()
        obs.update.side_effect = RuntimeError("boom")
        handle = _SpanHandle(obs)
        handle.update(key="value")  # should not raise


# ---------------------------------------------------------------------------
# create_request_trace
# ---------------------------------------------------------------------------


class TestCreateRequestTrace:
    def test_disabled_yields_noop(self):
        with create_request_trace(session_id="s1") as handle:
            assert isinstance(handle, _NoOpHandle)

    def test_enabled_yields_trace_handle(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_request_trace(session_id="s1", model="anthropic:claude-sonnet-4-5") as handle:
            assert isinstance(handle, _TraceHandle)

        mock_client.start_as_current_observation.assert_called_once()
        mock_obs.update_trace.assert_called_once()
        call_kwargs = mock_obs.update_trace.call_args[1]
        assert call_kwargs["session_id"] == "s1"
        assert call_kwargs["user_id"] == "elias"
        assert "model:anthropic:claude-sonnet-4-5" in call_kwargs["tags"]
        mock_cm.__exit__.assert_called_once()

    def test_tags_include_model_and_continuation(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_request_trace(
            session_id="s1",
            model="anthropic:claude-haiku-4-5",
            is_continuation=True,
            tags=["custom"],
        ):
            pass

        call_kwargs = mock_obs.update_trace.call_args[1]
        tags = call_kwargs["tags"]
        assert "custom" in tags
        assert "model:anthropic:claude-haiku-4-5" in tags
        assert "continuation" in tags

    def test_no_model_no_continuation_tags(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_request_trace(session_id="s1"):
            pass

        call_kwargs = mock_obs.update_trace.call_args[1]
        assert call_kwargs["tags"] == []

    def test_start_failure_yields_noop(self):
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = RuntimeError("boom")

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_request_trace(session_id="s1") as handle:
            assert isinstance(handle, _NoOpHandle)

    def test_exit_failure_is_swallowed(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(side_effect=RuntimeError("exit boom"))
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        # Should not raise despite __exit__ failure
        with create_request_trace(session_id="s1"):
            pass

    def test_metadata_passed_through(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_request_trace(
            session_id="s1",
            input_message="hello",
            metadata={"has_images": True},
        ):
            pass

        call_kwargs = mock_obs.update_trace.call_args[1]
        assert call_kwargs["metadata"] == {"has_images": True}
        assert call_kwargs["input"] == "hello"


# ---------------------------------------------------------------------------
# create_span
# ---------------------------------------------------------------------------


class TestCreateSpan:
    def test_disabled_yields_noop(self):
        with create_span("test-span") as handle:
            assert isinstance(handle, _NoOpHandle)

    def test_enabled_yields_span_handle(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_span("my-span", input_data="hello", metadata={"key": "val"}) as handle:
            assert isinstance(handle, _SpanHandle)

        mock_client.start_as_current_observation.assert_called_once_with(
            as_type="span",
            name="my-span",
            input="hello",
            metadata={"key": "val"},
        )
        mock_cm.__exit__.assert_called_once()

    def test_start_failure_yields_noop(self):
        mock_client = MagicMock()
        mock_client.start_as_current_observation.side_effect = RuntimeError("boom")

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_span("test-span") as handle:
            assert isinstance(handle, _NoOpHandle)

    def test_exit_failure_is_swallowed(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(side_effect=RuntimeError("exit boom"))
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        # Should not raise despite __exit__ failure
        with create_span("test-span"):
            pass

    def test_span_handle_update_output(self):
        mock_client = MagicMock()
        mock_obs = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_obs)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_client.start_as_current_observation.return_value = mock_cm

        tracing._tracing_enabled = True
        tracing._langfuse_client = mock_client

        with create_span("test-span") as handle:
            handle.update_output({"result": "ok"})

        mock_obs.update.assert_called_once_with(output={"result": "ok"})
