"""Configuration for the access module."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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
        default=["*"],
        description="Allowed CORS origins. Restrict this whenever the server is exposed.",
    )
