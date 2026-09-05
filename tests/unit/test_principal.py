"""Principals, credentials and the session ownership rule."""

from __future__ import annotations

import pytest

from assistant_runtime.principal import (
    LOCAL_PRINCIPAL,
    Credentials,
    Principal,
    can_access_session,
)


class TestPrincipal:
    def test_roles_are_a_frozenset_and_admin_is_a_role(self):
        principal = Principal(id="u1", roles=["viewer", "admin"], label="Ada")
        assert principal.roles == frozenset({"viewer", "admin"})
        assert principal.is_admin
        assert principal.to_dict() == {"id": "u1", "roles": ["admin", "viewer"], "label": "Ada"}
        assert not Principal(id="u2").is_admin

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_id_must_not_be_empty(self, bad):
        with pytest.raises(ValueError, match="must not be empty"):
            Principal(id=bad)

    def test_local_operator_is_an_admin(self):
        assert LOCAL_PRINCIPAL.id == "local"
        assert LOCAL_PRINCIPAL.is_admin


class TestCredentials:
    def test_headers_are_lower_cased(self):
        credentials = Credentials.from_headers(
            "http", {"X-Assistant-Principal": "u1", "Authorization": "Bearer t"}, client="10.0.0.1"
        )
        assert credentials.header("x-assistant-principal") == "u1"
        assert credentials.header("AUTHORIZATION") == "Bearer t"
        assert credentials.client == "10.0.0.1"
        assert credentials.transport == "http"

    def test_environ_headers_and_auth_payload(self):
        environ = {
            "HTTP_X_ASSISTANT_PRINCIPAL": "u2",
            "HTTP_COOKIE": "a=b",
            "REMOTE_ADDR": "127.0.0.1",
            "PATH_INFO": "/socket.io/",
        }
        credentials = Credentials.from_environ(environ, auth={"token": "t"})
        assert credentials.transport == "socketio"
        assert credentials.header("x-assistant-principal") == "u2"
        assert credentials.header("cookie") == "a=b"
        assert credentials.auth == {"token": "t"}
        assert credentials.client == "127.0.0.1"
        assert credentials.header("path-info") is None


class TestOwnership:
    def test_rules(self):
        owner = Principal(id="u1")
        other = Principal(id="u2")
        admin = Principal(id="root", roles={"admin"})
        assert can_access_session(owner, "u1")
        assert not can_access_session(other, "u1")
        assert can_access_session(admin, "u1")
        # Unowned (legacy or system-created) sessions are reachable until assigned.
        assert can_access_session(other, None)
