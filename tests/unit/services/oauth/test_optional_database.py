"""OAuth remains usable when the optional database cannot persist tokens."""

import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from cryptography.fernet import Fernet

from assistant_runtime.services.oauth.config import OAuthConfig
from assistant_runtime.services.oauth.interface import CodexCliAuth, OAuthService


@pytest.mark.parametrize("healthy", [False, True])
async def test_sync_survives_unavailable_database_and_reports_memory_only(healthy):
    @asynccontextmanager
    async def unavailable():
        if not healthy:
            pytest.fail("An unhealthy database must not be accessed")
        raise ConnectionError("database went down after its health check")
        yield  # pragma: no cover

    service = OAuthService(OAuthConfig(encryption_key=Fernet.generate_key().decode()))
    service.set_database_service(SimpleNamespace(healthy=healthy, session_context=unavailable))
    await service.start()
    try:
        with patch.object(
            service,
            "_read_codex_cli_auth",
            return_value=CodexCliAuth(
                access_token="test-access", refresh_token="test-refresh", account_id="test-account"
            ),
        ):
            status = await service.sync_from_codex_cli()
        assert status.connected
        assert status.source == "codex_cli"
        assert status.persisted is False
        assert service.get_codex_session().access_token == "test-access"
        if not healthy:
            assert await service.disconnect() is False
            assert not service.get_device_code_status().connected
    finally:
        await service.stop()


@pytest.mark.parametrize("row_existed", [True, False])
async def test_disconnect_reports_incomplete_deletion_then_retries_on_recovery(row_existed):
    session = SimpleNamespace(commit=AsyncMock())

    @asynccontextmanager
    async def session_context():
        yield session
        await session.commit()

    database = SimpleNamespace(healthy=False, session_context=session_context)
    service = OAuthService(OAuthConfig())
    service.set_database_service(database)
    service._access_token = "test-access"
    service._account_id = "test-account"
    service._token_persisted = True
    assert await service.disconnect() is False
    assert service.get_codex_session() is None
    database.healthy = True
    with patch("assistant_runtime.services.database.repositories.OAuthTokenRepository") as repo:
        repo.return_value.delete = AsyncMock(return_value=row_existed)
        assert await service.disconnect() is True
        repo.return_value.delete.assert_awaited_once_with("openai")
    session.commit.assert_awaited_once()


async def test_successful_database_save_is_encrypted_and_reported():
    session = SimpleNamespace(commit=AsyncMock())

    @asynccontextmanager
    async def session_context():
        yield session
        await session.commit()

    service = OAuthService(OAuthConfig(encryption_key=Fernet.generate_key().decode()))
    await service.start()
    service.set_database_service(SimpleNamespace(healthy=True, session_context=session_context))
    service._access_token = "test-access"
    service._account_id = "test-account"
    try:
        with patch("assistant_runtime.services.database.repositories.OAuthTokenRepository") as repo:
            repo.return_value.upsert = AsyncMock()
            await service._save_token()
            saved = repo.return_value.upsert.call_args.kwargs
        assert service.get_device_code_status().persisted
        assert saved["encrypted_api_key"] != "test-access"
        assert service._decrypt(saved["encrypted_api_key"]) == "test-access"
        session.commit.assert_awaited_once()
    finally:
        await service.stop()


async def test_codex_cli_sync_removes_only_a_stored_login():
    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    service = OAuthService(OAuthConfig(encryption_key=Fernet.generate_key().decode()))
    await service.start()
    service.set_database_service(SimpleNamespace(healthy=True, session_context=session_context))
    try:
        with (
            patch.object(
                service,
                "_read_codex_cli_auth",
                return_value=CodexCliAuth(
                    access_token="test-access",
                    refresh_token="test-refresh",
                    account_id="test-account",
                ),
            ),
            patch("assistant_runtime.services.database.repositories.OAuthTokenRepository") as repo,
        ):
            repo.return_value.upsert = AsyncMock()
            repo.return_value.delete = AsyncMock()
            repo.return_value.delete_login = AsyncMock(return_value=True)
            status = await service.sync_from_codex_cli()
        assert status.connected
        assert status.persisted is False
        # A stored API key shares the row; only the login-only delete may touch it.
        repo.return_value.delete_login.assert_awaited_once_with("openai")
        repo.return_value.delete.assert_not_awaited()
        repo.return_value.upsert.assert_not_awaited()
    finally:
        await service.stop()


async def test_a_stored_copy_of_the_codex_cli_login_is_handed_back_to_the_cli():
    fernet_key = Fernet.generate_key().decode()
    cipher = Fernet(fernet_key.encode())

    @asynccontextmanager
    async def session_context():
        yield SimpleNamespace()

    stored = SimpleNamespace(
        encrypted_api_key=cipher.encrypt(b"old-access").decode(),
        encrypted_refresh_token=cipher.encrypt(b"cli-refresh").decode(),
        encrypted_id_token=None,
        expires_at=time.time() + 60,
        email=None,
    )
    service = OAuthService(OAuthConfig(encryption_key=fernet_key))
    service.set_database_service(SimpleNamespace(healthy=True, session_context=session_context))
    with (
        patch.object(
            service,
            "_read_codex_cli_auth",
            return_value=CodexCliAuth(
                access_token="cli-access", refresh_token="cli-refresh", account_id="acct"
            ),
        ),
        patch("assistant_runtime.services.database.repositories.OAuthTokenRepository") as repo,
        patch.object(service, "_refresh_token_chain", AsyncMock()) as refresh,
    ):
        repo.return_value.get = AsyncMock(return_value=stored)
        repo.return_value.delete_login = AsyncMock(return_value=True)
        await service.start()
        try:
            status = service.get_device_code_status()
            assert status.source == "codex_cli"
            assert service._access_token == "cli-access"
            repo.return_value.delete_login.assert_awaited_once_with("openai")
            refresh.assert_not_awaited()
        finally:
            await service.stop()
