"""The local-install security defaults: browser origins, the local token, health detail.

These are the rules a page on another site, a probe and a reverse proxy
hit; docs/access.md describes them.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.access import deps as access_deps
from assistant_runtime.app.access.config import LOCALHOST_ORIGIN_REGEX, AccessConfig
from assistant_runtime.app.access.exceptions import AuthenticationError
from assistant_runtime.app.access.interface import AccessService
from assistant_runtime.app.socketio_server import create_sio
from assistant_runtime.base.lifecycle import LifecycleManager
from assistant_runtime.config import AppSettings
from assistant_runtime.main import create_app, public_health
from assistant_runtime.principal import Credentials


class TestOriginRule:
    def test_default_admits_this_machine_on_any_port(self):
        config = AccessConfig()
        assert config.cors_origin_regex == LOCALHOST_ORIGIN_REGEX
        for origin in (
            "http://localhost:3000",
            "http://127.0.0.1:7140",
            "https://localhost",
            "http://[::1]:5173",
        ):
            assert config.origin_allowed(origin), origin
        for origin in ("https://evil.example", "http://localhost.evil.example", None, ""):
            assert not config.origin_allowed(origin), origin

    def test_explicit_origins_and_the_wildcard(self):
        config = AccessConfig(cors_origins=["https://app.example.com"], cors_origin_regex=None)
        assert config.origin_allowed("https://app.example.com")
        assert not config.origin_allowed("http://localhost:3000")
        assert AccessConfig(cors_origins=["*"]).origin_allowed("https://anything.example")

    @pytest.mark.asyncio
    async def test_http_preflight_follows_the_rule(self):
        app = create_app(settings=AppSettings(_env_file=None))
        app.state.lifecycle = LifecycleManager()
        headers = {"Access-Control-Request-Method": "POST"}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            local = await c.options(
                "/api/chat", headers={**headers, "Origin": "http://localhost:3000"}
            )
            other = await c.options(
                "/api/chat", headers={**headers, "Origin": "https://evil.example"}
            )
        assert local.status_code == 200
        assert local.headers["access-control-allow-origin"] == "http://localhost:3000"
        assert other.status_code == 400
        assert "access-control-allow-origin" not in other.headers

    def test_socketio_applies_the_same_configuration(self):
        check = create_sio(AccessConfig()).eio.cors_allowed_origins
        assert callable(check)
        assert check("http://localhost:7140", {}) is True
        assert check("https://evil.example", {}) is False
        # The page's own origin is always allowed, like Engine.IO's default.
        environ = {"wsgi.url_scheme": "https", "HTTP_HOST": "runtime.example"}
        assert check("https://runtime.example", environ) is True
        assert check("https://other.example", environ) is False
        assert create_sio(AccessConfig(cors_origins=["*"])).eio.cors_allowed_origins == "*"


class TestLocalToken:
    @pytest.mark.asyncio
    async def test_unset_token_trusts_everyone(self):
        service = AccessService(AccessConfig())
        principal = await service.authenticate(Credentials.from_headers("http", {}))
        assert principal.is_admin

    @pytest.mark.asyncio
    async def test_bearer_header_or_socket_auth_must_match(self):
        service = AccessService(AccessConfig(local_token="s3cret"))
        with pytest.raises(AuthenticationError):
            await service.authenticate(Credentials.from_headers("http", {}))
        with pytest.raises(AuthenticationError):
            await service.authenticate(
                Credentials.from_headers("http", {"Authorization": "Bearer wrong"})
            )
        ok = await service.authenticate(
            Credentials.from_headers("http", {"Authorization": "Bearer s3cret"})
        )
        assert ok.is_admin
        socket = Credentials.from_environ({}, auth={"token": "s3cret"})
        assert (await service.authenticate(socket)).is_admin
        with pytest.raises(AuthenticationError):
            await service.authenticate(Credentials.from_environ({}, auth={"token": "nope"}))


class TestHealthDetail:
    @pytest.mark.asyncio
    async def test_anonymous_callers_get_flags_only(self, app, client, monkeypatch):
        """In header mode a caller without the proxy header is anonymous."""
        component = _component({"healthy": True, "providers": ["anthropic"], "host": "db"})
        await app.state.lifecycle.register("llm_service", component)
        monkeypatch.setattr(
            access_deps,
            "get_access_service",
            lambda request: AccessService(AccessConfig(mode="header")),
        )
        anonymous = (await client.get("/health")).json()
        assert anonymous["runtime"] == "assistant-runtime"
        assert anonymous["components"] == {"llm_service": {"healthy": True}}
        detailed = (await client.get("/health", headers={"X-Assistant-Principal": "alice"})).json()
        assert detailed["components"]["llm_service"]["providers"] == ["anthropic"]

    @pytest.mark.asyncio
    async def test_the_local_operator_sees_the_detail(self, app, client):
        component = _component({"healthy": True, "primary_model": "anthropic:claude-opus-5"})
        await app.state.lifecycle.register("llm_service", component)
        data = (await client.get("/health")).json()
        assert data["components"]["llm_service"]["primary_model"] == "anthropic:claude-opus-5"

    def test_public_projection_keeps_only_flags(self):
        result = {
            "healthy": False,
            "components": {"a": {"healthy": True, "secret": 1}, "b": {"healthy": False}},
        }
        assert public_health(result) == {
            "healthy": False,
            "runtime": "assistant-runtime",
            "components": {"a": {"healthy": True}, "b": {"healthy": False}},
        }


class TestAuthenticatedRoutes:
    @pytest.mark.asyncio
    async def test_models_and_media_need_a_principal(self, app, client, monkeypatch):
        from unittest.mock import MagicMock

        app.state.media_service = MagicMock(get_cached_image=lambda image_id: None)
        monkeypatch.setattr(
            access_deps,
            "get_access_service",
            lambda request: AccessService(AccessConfig(mode="header")),
        )
        assert (await client.get("/api/models")).status_code == 401
        assert (await client.get("/api/media/abc")).status_code == 401
        assert (await client.get("/api/media/video/abc")).status_code == 401
        header = {"X-Assistant-Principal": "alice"}
        assert (await client.get("/api/models", headers=header)).status_code == 200
        assert (await client.get("/api/media/abc", headers=header)).status_code == 404


def _component(health: dict):
    from unittest.mock import AsyncMock

    component = AsyncMock()
    component.health_check = AsyncMock(return_value=health)
    return component
