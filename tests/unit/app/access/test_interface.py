"""AccessService: the three authentication modes and the shared rules."""

from __future__ import annotations

import pytest

from assistant_runtime.app.access.config import AccessConfig
from assistant_runtime.app.access.exceptions import AccessDeniedError, AuthenticationError
from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.principal import LOCAL_PRINCIPAL, Credentials, Principal


def _credentials(**headers: str) -> Credentials:
    return Credentials.from_headers("http", headers)


class TestTrustedLocal:
    async def test_everyone_is_the_local_operator(self):
        service = AccessService(AccessConfig())
        await service.start()
        assert (await service.health_check()) == {"healthy": True, "mode": "trusted_local"}
        assert await service.authenticate(_credentials()) is LOCAL_PRINCIPAL
        assert (
            await service.authenticate(_credentials(x_assistant_principal="u1")) is LOCAL_PRINCIPAL
        )


class TestHeaderMode:
    @pytest.fixture
    def service(self):
        return AccessService(AccessConfig(mode="header"))

    async def test_principal_and_roles_come_from_the_proxy_headers(self, service):
        principal = await service.authenticate(
            Credentials.from_headers(
                "http", {"X-Assistant-Principal": " u1 ", "X-Assistant-Roles": "admin, ops ,"}
            )
        )
        assert principal == Principal(id="u1", roles={"admin", "ops"})

    async def test_missing_header_is_unauthenticated(self, service):
        with pytest.raises(AuthenticationError, match="X-Assistant-Principal"):
            await service.authenticate(_credentials())
        with pytest.raises(AuthenticationError):
            await service.authenticate(
                Credentials.from_headers("http", {"X-Assistant-Principal": "  "})
            )

    async def test_header_names_are_configurable(self):
        service = AccessService(
            AccessConfig(mode="header", principal_header="X-User", roles_header="X-Groups")
        )
        principal = await service.authenticate(
            Credentials.from_headers("socketio", {"x-user": "u9", "x-groups": "admin"})
        )
        assert principal.id == "u9"
        assert principal.is_admin


class TestHostMode:
    async def test_sync_and_async_callbacks(self):
        def by_token(credentials: Credentials) -> Principal | None:
            token = credentials.header("authorization")
            return Principal(id="u1") if token == "Bearer ok" else None

        async def by_token_async(credentials: Credentials) -> Principal | None:
            return by_token(credentials)

        for callback in (by_token, by_token_async):
            service = AccessService(AccessConfig(mode="host"), authenticator=callback)
            await service.start()
            principal = await service.authenticate(_credentials(authorization="Bearer ok"))
            assert principal.id == "u1"
            with pytest.raises(AuthenticationError, match="Not authenticated"):
                await service.authenticate(_credentials(authorization="Bearer no"))

    async def test_callback_errors_and_wrong_types_are_unauthenticated(self):
        def boom(credentials):
            raise RuntimeError("identity provider down")

        service = AccessService(AccessConfig(mode="host"), authenticator=boom)
        with pytest.raises(AuthenticationError, match="Authentication failed"):
            await service.authenticate(_credentials())

        service = AccessService(AccessConfig(mode="host"), authenticator=lambda c: "u1")
        with pytest.raises(AuthenticationError, match="no Principal"):
            await service.authenticate(_credentials())

    async def test_callback_may_raise_authentication_error_itself(self):
        def strict(credentials):
            raise AuthenticationError("token expired")

        service = AccessService(AccessConfig(mode="host"), authenticator=strict)
        with pytest.raises(AuthenticationError, match="token expired"):
            await service.authenticate(_credentials())

    async def test_host_mode_needs_a_callback_to_start(self):
        service = AccessService(AccessConfig(mode="host"))
        with pytest.raises(AccessDeniedError, match="AssistantDefinition.authenticate"):
            await service.start()


class TestRules:
    def test_require_admin(self):
        AccessService.require_admin(Principal(id="a", roles={"admin"}))
        with pytest.raises(AccessDeniedError, match="admin role"):
            AccessService.require_admin(Principal(id="u"))

    def test_check_session(self):
        AccessService.check_session(Principal(id="u1"), "u1", "s")
        AccessService.check_session(Principal(id="u1"), None, "s")
        with pytest.raises(AccessDeniedError, match="Session 's' belongs to another principal"):
            AccessService.check_session(Principal(id="u2"), "u1", "s")
