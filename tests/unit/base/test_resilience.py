"""Tests for retry_with_backoff decorator from base.resilience."""

from unittest.mock import patch

import pytest

from assistant_runtime.base.resilience import retry_with_backoff


class TestRetryWithBackoff:
    """Tests for the retry_with_backoff decorator."""

    def test_retries_on_specified_exception_then_succeeds(self):
        """Transient failure followed by success — should return the successful result."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(ValueError,),
            name="transient_test",
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        result = flaky()
        assert result == "ok"
        assert call_count == 3

    def test_gives_up_after_max_attempts(self):
        """Should re-raise the original exception after exhausting all attempts."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(ValueError,),
            name="give_up_test",
        )
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError, match="permanent"):
            always_fail()

        assert call_count == 3

    def test_does_not_retry_on_non_matching_exception(self):
        """Non-matching exception types should pass through immediately without retry."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(ConnectionError,),
            name="no_match_test",
        )
        def wrong_exception():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError, match="not retryable"):
            wrong_exception()

        # Should have been called exactly once -- no retries for non-matching exception
        assert call_count == 1

    def test_logging_callback_fires_on_each_retry(self):
        """The before_sleep callback should log on each retry attempt."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(ValueError,),
            name="log_test",
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return "ok"

        with patch("assistant_runtime.base.resilience.logger") as mock_logger:
            flaky()

        # 2 retries = 2 before_sleep calls = 2 warning logs
        assert mock_logger.warning.call_count == 2

    def test_name_parameter_appears_in_log_output(self):
        """The name kwarg should be passed to the logger as structured context."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=2,
            min_wait=0,
            max_wait=0,
            retry_on=(ValueError,),
            name="my_operation",
        )
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("oops")
            return "ok"

        with patch("assistant_runtime.base.resilience.logger") as mock_logger:
            flaky()

        mock_logger.warning.assert_called_once()
        _args, kwargs = mock_logger.warning.call_args
        assert kwargs["name"] == "my_operation"

    def test_default_retry_on_includes_connection_and_timeout_error(self):
        """Default retry_on=(ConnectionError, TimeoutError) should retry on those types."""
        call_count = 0

        @retry_with_backoff(max_attempts=2, min_wait=0, max_wait=0, name="defaults")
        def connection_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("refused")

        with pytest.raises(ConnectionError):
            connection_fail()

        # Should have retried (2 attempts total)
        assert call_count == 2

    def test_default_retry_on_timeout_error(self):
        """Default retry_on should also handle TimeoutError."""
        call_count = 0

        @retry_with_backoff(max_attempts=2, min_wait=0, max_wait=0, name="timeout_defaults")
        def timeout_fail():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("timed out")
            return "recovered"

        result = timeout_fail()
        assert result == "recovered"
        assert call_count == 2

    def test_custom_retry_on_tuple_is_respected(self):
        """Custom retry_on should only retry on specified exception types."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(KeyError,),
            name="custom_retry",
        )
        def key_error_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise KeyError("missing")
            return "found"

        result = key_error_then_ok()
        assert result == "found"
        assert call_count == 2

    def test_max_attempts_one_means_no_retries(self):
        """With max_attempts=1, the function is called once and the exception is re-raised."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=1,
            min_wait=0,
            max_wait=0,
            retry_on=(ValueError,),
            name="no_retry",
        )
        def fail_once():
            nonlocal call_count
            call_count += 1
            raise ValueError("one shot")

        with pytest.raises(ValueError, match="one shot"):
            fail_once()

        assert call_count == 1

    async def test_async_function_works_with_decorator(self):
        """The tenacity decorator should work with async functions."""
        call_count = 0

        @retry_with_backoff(
            max_attempts=3,
            min_wait=0,
            max_wait=0,
            retry_on=(ConnectionError,),
            name="async_test",
        )
        async def async_flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("async transient")
            return "async_ok"

        result = await async_flaky()
        assert result == "async_ok"
        assert call_count == 3

    def test_preserves_original_exception_type_on_reraise(self):
        """When max_attempts is exhausted, the original exception type should be re-raised."""

        class CustomDomainError(Exception):
            pass

        @retry_with_backoff(
            max_attempts=2,
            min_wait=0,
            max_wait=0,
            retry_on=(CustomDomainError,),
            name="preserve_type",
        )
        def always_custom_fail():
            raise CustomDomainError("domain failure")

        with pytest.raises(CustomDomainError, match="domain failure"):
            always_custom_fail()
