"""Tests for OAuthService Codex CLI sync behavior."""

from __future__ import annotations

import asyncio
import base64
import json
import time

import httpx
import pytest
from cryptography.fernet import Fernet

from assistant_runtime.services.oauth.config import OAuthConfig
from assistant_runtime.services.oauth.exceptions import OAuthCodexSyncError
from assistant_runtime.services.oauth.interface import AuthSource, DeviceCodeStatus, OAuthService


def _make_jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.sig"


def _write_codex_auth(path, *, email: str = "assistant@example.com") -> None:
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "last_refresh": "2026-03-15T11:48:53Z",
                "tokens": {
                    "access_token": "access-token",
                    "refresh_token": "refresh-token",
                    "id_token": _make_jwt(
                        {
                            "email": email,
                            "https://api.openai.com/auth": {
                                "chatgpt_account_id": "acct_123",
                            },
                            "https://api.openai.com/profile": {
                                "email": email,
                            },
                        }
                    ),
                    "account_id": "acct_123",
                },
            }
        )
    )


class TestOAuthServiceCodexSync:
    @pytest.fixture(autouse=True)
    def clear_openai_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    @pytest.fixture
    def encryption_key(self) -> str:
        return Fernet.generate_key().decode()

    async def test_sync_from_codex_cli_imports_tokens_and_exposes_codex_session(
        self, tmp_path, encryption_key
    ):
        auth_file = tmp_path / "auth.json"
        _write_codex_auth(auth_file)

        service = OAuthService(
            OAuthConfig(
                encryption_key=encryption_key,
                codex_auth_file=str(auth_file),
                codex_auto_sync=False,
            )
        )
        await service.start()

        status = await service.sync_from_codex_cli()

        assert status.connected is True
        assert status.status == DeviceCodeStatus.AUTHORIZED
        assert status.source == AuthSource.CODEX_CLI
        assert status.email == "assistant@example.com"
        session = service.get_codex_session()
        assert session is not None
        assert session.account_id == "acct_123"
        await service.stop()

    async def test_start_auto_syncs_codex_cli_when_enabled(self, tmp_path, encryption_key):
        auth_file = tmp_path / "auth.json"
        _write_codex_auth(auth_file, email="auto@example.com")

        service = OAuthService(
            OAuthConfig(
                encryption_key=encryption_key,
                codex_auth_file=str(auth_file),
                codex_auto_sync=True,
            )
        )

        await service.start()

        status = service.get_device_code_status()
        assert status.connected is True
        assert status.source == AuthSource.CODEX_CLI
        assert status.email == "auto@example.com"
        session = service.get_codex_session()
        assert session is not None
        assert session.account_id == "acct_123"
        await service.stop()

    async def test_sync_from_codex_cli_raises_for_missing_file(self, tmp_path, encryption_key):
        missing = tmp_path / "missing.json"
        service = OAuthService(
            OAuthConfig(
                encryption_key=encryption_key,
                codex_auth_file=str(missing),
                codex_auto_sync=False,
            )
        )
        await service.start()

        with pytest.raises(OAuthCodexSyncError, match="Codex auth file not found"):
            await service.sync_from_codex_cli()

        await service.stop()


async def _service_with_transport(handler) -> OAuthService:
    service = OAuthService(
        OAuthConfig(encryption_key=Fernet.generate_key().decode(), codex_auto_sync=False)
    )
    await service.start()
    await service._http.aclose()
    service._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return service


@pytest.mark.parametrize("operation", ["disconnect", "replace", "stop"])
async def test_pending_device_flow_is_drained_before_authentication_changes(operation):
    polled, closed = asyncio.Event(), asyncio.Event()

    async def handler(request):
        if request.url.path.endswith("/usercode"):
            return httpx.Response(
                200, json={"interval": 0, "device_auth_id": "test-device", "user_code": "test-code"}
            )
        polled.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    service = await _service_with_transport(handler)
    try:
        await service.initiate_device_code()
        async with asyncio.timeout(5):
            await polled.wait()
            old_flow = service._poll_task
            if operation == "replace":
                await service.initiate_device_code()
            elif operation == "disconnect":
                assert await service.disconnect() is True
            else:
                await service.stop()
        assert old_flow.done()
        assert closed.is_set()
        assert service.get_codex_session() is None
        if operation == "replace":
            assert service._poll_task is not old_flow
        else:
            assert service._poll_task is None
    finally:
        await service.stop()


@pytest.mark.parametrize("waiting_on", ["provider", "persistence"])
async def test_disconnect_drains_refresh_before_clearing_credentials(waiting_on, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    order = []

    async def wait_if(phase):
        if waiting_on == phase:
            entered.set()
            await release.wait()

    async def handler(request):
        await wait_if("provider")
        return httpx.Response(
            200,
            json={
                "access_token": _make_jwt(
                    {
                        "exp": time.time() + 7200,
                        "https://api.openai.com/auth": {"chatgpt_account_id": "test-account"},
                    }
                ),
                "refresh_token": "test-refreshed-token",
            },
        )

    async def save():
        await wait_if("persistence")
        order.append("saved")

    service = await _service_with_transport(handler)
    service._refresh_token = "test-refresh-token"
    monkeypatch.setattr(service, "_save_token", save)
    refresh = asyncio.create_task(service.refresh())
    disconnect = None
    try:
        async with asyncio.timeout(5):
            await entered.wait()
            disconnect = asyncio.create_task(service.disconnect())
            await asyncio.sleep(0)
            assert not disconnect.done()
            release.set()
            await refresh
            assert await disconnect is True
        assert order == ["saved"]
        assert service.get_codex_session() is None
        assert service._refresh_token is None
        assert service.get_device_code_status().status == DeviceCodeStatus.IDLE
    finally:
        release.set()
        await asyncio.gather(refresh, *([disconnect] if disconnect else []), return_exceptions=True)
        await service.stop()
