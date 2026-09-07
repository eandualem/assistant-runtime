"""Who is acting: the trusted principal of a request, and what it may reach.

A leaf module (no project imports) shared by the app layer, the tools
layer and host code. A principal is established by the host's
authentication (see ``app/access``), never by the request body: nothing a
client sends in a message can change it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

ADMIN_ROLE = "admin"
Transport = Literal["http", "socketio", "in_process"]


@dataclass(frozen=True)
class Principal:
    """A trusted identity. ``roles`` containing ``admin`` unlocks administration."""

    id: str
    roles: frozenset[str] = frozenset()
    label: str = ""

    def __post_init__(self) -> None:
        if not self.id or not self.id.strip():
            raise ValueError("Principal id must not be empty")
        object.__setattr__(self, "roles", frozenset(self.roles))

    @property
    def is_admin(self) -> bool:
        return ADMIN_ROLE in self.roles

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "roles": sorted(self.roles), "label": self.label}


LOCAL_PRINCIPAL = Principal(id="local", roles=frozenset({ADMIN_ROLE}), label="local operator")
"""The single operator of a trusted local installation, and the identity of
in-process callers that do not name one."""


@dataclass(frozen=True)
class Credentials:
    """What a transport knows about the caller, for the authentication hook.

    ``headers`` keys are lower-case. ``auth`` is the Socket.IO connect
    payload (whatever the client put there); ``client`` the peer address.
    """

    transport: Transport
    headers: Mapping[str, str] = field(default_factory=dict)
    auth: Any = None
    client: str | None = None

    @classmethod
    def from_headers(
        cls,
        transport: Transport,
        headers: Mapping[str, str],
        *,
        auth: Any = None,
        client: str | None = None,
    ) -> Credentials:
        return cls(
            transport=transport,
            headers={str(k).lower(): str(v) for k, v in headers.items()},
            auth=auth,
            client=client,
        )

    @classmethod
    def from_environ(cls, environ: Mapping[str, Any], auth: Any = None) -> Credentials:
        """From a WSGI-style environ (python-socketio passes one to ``on_connect``)."""
        headers = {
            key[5:].replace("_", "-").lower(): str(value)
            for key, value in environ.items()
            if isinstance(key, str) and key.startswith("HTTP_")
        }
        client = environ.get("REMOTE_ADDR")
        return cls(
            transport="socketio",
            headers=headers,
            auth=auth,
            client=str(client) if client else None,
        )

    def header(self, name: str) -> str | None:
        return self.headers.get(name.lower())


def can_access_session(principal: Principal, owner_id: str | None) -> bool:
    """Whether ``principal`` may read or act on a session owned by ``owner_id``.

    Admins reach every session. A session without an owner (created before
    ownership existed) is reachable only by administrators until one assigns
    it; ordinary principals reach their own sessions and nothing else.
    """
    return principal.is_admin or (owner_id is not None and owner_id == principal.id)


__all__ = [
    "ADMIN_ROLE",
    "LOCAL_PRINCIPAL",
    "Credentials",
    "Principal",
    "Transport",
    "can_access_session",
]
