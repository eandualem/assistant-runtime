"""Tests for the @instrument decorator."""

import pytest

from lovely_assistant.base.instrument import instrument


async def test_instrument_returns_result():
    @instrument(operation="test_op", module="test")
    async def my_func():
        return 42

    assert await my_func() == 42


async def test_instrument_reraises_exceptions():
    @instrument(operation="failing_op", module="test")
    async def failing_func():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await failing_func()


async def test_instrument_logs_success(capfd):
    @instrument(operation="logged_op", module="test_mod")
    async def my_func():
        return "ok"

    await my_func()
    # The decorator logs via loguru — we verify it doesn't error.
    # Detailed log content testing would require loguru sink capture.


async def test_instrument_defaults_to_function_name():
    @instrument(module="test")
    async def specific_name():
        return True

    # Verify the wrapper preserves the function name
    assert specific_name.__name__ == "specific_name"
    assert await specific_name() is True


async def test_instrument_preserves_args():
    @instrument(operation="echo", module="test")
    async def echo(a, b, key=None):
        return (a, b, key)

    result = await echo(1, 2, key="val")
    assert result == (1, 2, "val")
