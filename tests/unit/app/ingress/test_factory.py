"""Tests for the ingress factory: it wires the service into streaming."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from assistant_runtime.app.ingress.factory import register_ingress
from assistant_runtime.app.ingress.interface import IngressService


async def test_register_ingress_attaches_the_service_to_streaming():
    streaming = SimpleNamespace(attach_ingress=MagicMock())
    app_state = SimpleNamespace(
        assistant_service=MagicMock(), streaming_service=streaming, database_service=None, sio=None
    )
    lifecycle = MagicMock()
    lifecycle.register = AsyncMock()

    await register_ingress(app_state, lifecycle)

    assert isinstance(app_state.ingress_service, IngressService)
    streaming.attach_ingress.assert_called_once_with(app_state.ingress_service)
    lifecycle.register.assert_awaited_once_with("ingress_service", app_state.ingress_service)
