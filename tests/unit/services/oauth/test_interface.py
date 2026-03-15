"""Tests for OAuthService Codex CLI sync behavior."""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.fernet import Fernet

from lovely_assistant.services.oauth.config import OAuthConfig
from lovely_assistant.services.oauth.exceptions import OAuthCodexSyncError
from lovely_assistant.services.oauth.interface import AuthSource, DeviceCodeStatus, OAuthService


def _make_jwt(payload: dict[str, object]) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{header}.{body}.sig"


def _write_codex_auth(path, *, email: str = "jarvis@example.com") -> None:
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
        assert status.email == "jarvis@example.com"
        session = service.get_codex_session()
        assert session is not None
        assert session.account_id == "acct_123"
        assert service.get_openai_api_key() is None
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
