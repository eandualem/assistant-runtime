"""Configuration for the access module."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
"""Browser origins on this machine, any port: the development default."""


class AccessConfig(BaseModel):
    """Access module configuration. Nested into AppSettings as `access`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["trusted_local", "header", "host"] = Field(
        default="trusted_local",
        description=(
            "trusted_local: every caller is the local operator (admin); bind to localhost. "
            "header: a reverse proxy that authenticated the caller sets the principal header. "
            "host: the AssistantDefinition.authenticate callback decides."
        ),
    )
    principal_header: str = Field(
        default="X-Assistant-Principal",
        description="Header carrying the principal id in header mode.",
    )
    roles_header: str = Field(
        default="X-Assistant-Roles",
        description="Header carrying comma-separated roles in header mode; 'admin' administers.",
    )
    cors_origins: list[str] = Field(
        default_factory=list,
        description=(
            "Browser origins allowed exactly, for HTTP and Socket.IO. '*' allows every "
            "origin (credentials are then refused); an empty list relies on cors_origin_regex."
        ),
    )
    cors_origin_regex: str | None = Field(
        default=LOCALHOST_ORIGIN_REGEX,
        description=(
            "Regular expression for allowed origins. The default admits localhost on any "
            "port, so a page on another host cannot call the runtime. Unset it (empty) when "
            "exposing the server and list the real origins in cors_origins."
        ),
    )
    local_token: str = Field(
        default="",
        description=(
            "trusted_local only. When set, every caller must present it as "
            "'Authorization: Bearer <token>' (HTTP) or as auth.token (Socket.IO connect); "
            "a request without it is unauthenticated. Empty (the default) trusts every caller."
        ),
    )

    def origin_allowed(self, origin: str | None) -> bool:
        """Whether a browser ``Origin`` may reach the runtime under this configuration."""
        if not origin:
            return False
        if "*" in self.cors_origins or origin in self.cors_origins:
            return True
        if self.cors_origin_regex:
            return re.fullmatch(self.cors_origin_regex, origin) is not None
        return False
