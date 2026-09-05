"""The host contract: what an application tells the runtime about itself.

A leaf module (no project imports) shared by the app layer and the tools
layer. Version 1 of the structures a host sends with a message (or when joining a
session). Everything here is *application context*: it shapes the prompt
and the tools of a turn. None of it is identity or authorization — the
``host`` block is descriptive metadata, and a request cannot widen what it
may do by describing itself differently.

The wire form accepts camelCase or snake_case keys and the legacy ``page``
name for ``view``; the canonical stored form is ``HostContext.to_dict()``
(snake_case, ``None`` values dropped). Unknown fields are rejected so
mistakes surface at the edge; deliberate host-specific data goes under
``extensions``.

Size rules (characters of the JSON form): ``MAX_CONTEXT_CHARS`` for the
``view.data`` + ``view.state`` + ``extensions`` payload, and
``MAX_ATTACHMENT_CHARS`` for one attachment's data URI or text; at most
``MAX_NAVIGATION`` targets, ``MAX_ACTIONS`` actions and ``MAX_ATTACHMENTS``
attachments per context. Keys inside ``view.data``, ``view.state``,
``background``, ``extensions`` and action ``parameters`` are the host's own
and are never renamed.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HOST_CONTEXT_VERSION = 1
MAX_CONTEXT_CHARS = 32_000
MAX_ATTACHMENT_CHARS = 12_000_000
MAX_NAVIGATION = 50
MAX_ACTIONS = 32
MAX_ATTACHMENTS = 16
# What providers accept as a tool name; request-declared actions become tools.
ACTION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_CAMEL_RE_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def camel_to_snake(name: str) -> str:
    """``eventType`` → ``event_type``; ``HTMLParser`` → ``html_parser``; snake stays."""
    return _CAMEL_RE_2.sub(r"\1_\2", _CAMEL_RE_1.sub(r"\1_\2", name)).lower()


# Host-owned payloads the runtime passes through verbatim: renaming keys in
# them would change the host's data or a JSON schema's property names.
_OPAQUE_KEYS = frozenset({"data", "state", "background", "extensions", "parameters"})


def _snake_keys(obj: Any) -> Any:
    """Normalise the contract's own field names; opaque host payloads stay as sent."""
    if isinstance(obj, dict):
        return {
            camel_to_snake(k): (v if camel_to_snake(k) in _OPAQUE_KEYS else _snake_keys(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_snake_keys(item) for item in obj]
    return obj


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HostIdentity(_Strict):
    """Descriptive metadata about the client. Never trusted for authorization."""

    name: str = Field(min_length=1, max_length=64)
    kind: Literal["browser", "mobile", "desktop", "terminal", "service", "other"] = "other"
    version: str | None = Field(default=None, max_length=32)


class View(_Strict):
    """What the host is showing: a page, a screen, a document, a terminal pane."""

    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    """Curated content the model may reason about; rendered as key-value pairs."""
    state: dict[str, Any] = Field(default_factory=dict)
    """Selection, filters, workflow state; passed to the model as JSON."""


class NavigationTarget(_Strict):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    route: str | None = Field(default=None, max_length=512)
    """A host-side locator (URL, screen id); kept for the host, not rendered."""


class HostAction(_Strict):
    """An action the host performs when the model calls it, declared for this turn.

    It becomes a host tool with the same call/continuation protocol as the
    ones declared in configuration; a name that is already a registered tool
    is ignored with a warning.
    """

    name: str
    description: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("name")
    @classmethod
    def _valid_name(cls, value: str) -> str:
        if not ACTION_NAME_RE.fullmatch(value):
            raise ValueError(f"action name {value!r} must match {ACTION_NAME_RE.pattern}")
        return value

    @field_validator("parameters")
    @classmethod
    def _object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type", "object") != "object":
            raise ValueError("action parameters must be a JSON schema of type object")
        return value


class Attachment(_Strict):
    """Content the host attaches to a message.

    Exactly one of ``data_uri``, ``url`` or ``text`` carries it. ``purpose``
    ``reference`` (default) puts it in the model's message as native content;
    ``screenshot`` keeps it out of the message until the model calls
    ``look_at_screen``.
    """

    kind: Literal["image", "document", "text"] = "image"
    purpose: Literal["reference", "screenshot"] = "reference"
    name: str | None = Field(default=None, max_length=128)
    description: str | None = Field(default=None, max_length=512)
    media_type: str | None = Field(default=None, max_length=64)
    data_uri: str | None = None
    url: str | None = Field(default=None, max_length=2048)
    text: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> Attachment:
        sources = [s for s in (self.data_uri, self.url, self.text) if s is not None]
        if len(sources) != 1:
            raise ValueError("an attachment needs exactly one of data_uri, url or text")
        if self.data_uri is not None:
            if not self.data_uri.startswith("data:"):
                raise ValueError("data_uri must be a data: URI")
            if len(self.data_uri) > MAX_ATTACHMENT_CHARS:
                raise ValueError(f"attachment exceeds {MAX_ATTACHMENT_CHARS} characters")
        if self.text is not None:
            if self.kind != "text":
                raise ValueError("text attachments must have kind 'text'")
            if len(self.text) > MAX_ATTACHMENT_CHARS:
                raise ValueError(f"attachment exceeds {MAX_ATTACHMENT_CHARS} characters")
        elif self.kind == "text":
            raise ValueError("kind 'text' needs 'text'")
        if self.purpose == "screenshot" and (self.kind != "image" or self.data_uri is None):
            raise ValueError("a screenshot must be an image carried as a data_uri")
        return self

    @classmethod
    def screenshot(cls, data_uri: str) -> Attachment:
        return cls(kind="image", purpose="screenshot", data_uri=data_uri)


class HostContext(_Strict):
    """Version 1 of the host contract. See the module docstring."""

    version: int = HOST_CONTEXT_VERSION
    host: HostIdentity | None = None
    view: View | None = None
    navigation: list[NavigationTarget] = Field(default_factory=list, max_length=MAX_NAVIGATION)
    background: dict[str, Any] = Field(default_factory=dict)
    """Summaries of what is off screen: ``{"name": {"state": ..., "summary": {...}}}``."""
    actions: list[HostAction] = Field(default_factory=list, max_length=MAX_ACTIONS)
    attachments: list[Attachment] = Field(default_factory=list, max_length=MAX_ATTACHMENTS)
    captured_at: datetime | None = None
    """When the host captured this context; rendered so the model can judge freshness."""
    extensions: dict[str, Any] = Field(default_factory=dict)
    """Host-specific data the runtime does not interpret; shown to the model as JSON."""

    @field_validator("version")
    @classmethod
    def _supported_version(cls, value: int) -> int:
        if value != HOST_CONTEXT_VERSION:
            raise ValueError(
                f"host_context version {value} is not supported; this runtime speaks "
                f"version {HOST_CONTEXT_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def _within_size(self) -> HostContext:
        payload = {
            "data": self.view.data if self.view else {},
            "state": self.view.state if self.view else {},
            "extensions": self.extensions,
        }
        size = len(json.dumps(payload, default=str))
        if size > MAX_CONTEXT_CHARS:
            raise ValueError(
                f"host_context data/state/extensions total {size} characters; the limit is "
                f"{MAX_CONTEXT_CHARS}. Curate what the model needs instead of sending everything."
            )
        names = [action.name for action in self.actions]
        if len(names) != len(set(names)):
            raise ValueError("host_context.actions contains duplicate names")
        return self

    # --- wire adapters ---

    @classmethod
    def from_payload(cls, raw: Any) -> HostContext | None:
        """Parse the wire form: camelCase or snake_case, ``page`` alias, legacy shapes.

        Returns None for anything that is not a mapping. Raises ``ValueError``
        (through pydantic) with the field path for invalid content.
        """
        if not isinstance(raw, dict):
            return None
        data = _snake_keys(raw)
        if "page" in data:
            if "view" in data:
                raise ValueError("host_context carries both 'page' and 'view'; send one")
            data["view"] = data.pop("page")
        view = data.get("view")
        if isinstance(view, dict) and "actions" in view:
            raise ValueError(
                "host_context.page.actions is no longer rendered as text; declare callable "
                "actions at host_context.actions (name, description, parameters)"
            )
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """The canonical stored form."""
        return self.model_dump(mode="json", exclude_none=True)

    @property
    def view_name(self) -> str | None:
        return self.view.name if self.view else None

    def age_seconds(self, now: datetime | None = None) -> float | None:
        """How old the context is, when the host said when it captured it."""
        if self.captured_at is None:
            return None
        captured = self.captured_at
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=UTC)
        return ((now or datetime.now(UTC)) - captured).total_seconds()


def host_context_from_payload(data: Any) -> dict[str, Any] | None:
    """Pull and normalise the host context out of a raw client payload, if any.

    The canonical dict form, or None when the payload carries none.
    """
    if not isinstance(data, dict):
        return None
    for key in ("host_context", "hostContext"):
        if key in data:
            context = HostContext.from_payload(data[key])
            return context.to_dict() if context is not None else None
    return None


def view_name_of(host_context: dict[str, Any] | None) -> str | None:
    """The view name of a canonical host-context dict (``page`` accepted for older data)."""
    if not host_context:
        return None
    for key in ("view", "page"):
        view = host_context.get(key)
        if isinstance(view, dict) and view.get("name"):
            return str(view["name"])
    return None


def actions_of(host_context: dict[str, Any] | None) -> list[HostAction]:
    """The request-declared actions of a canonical host-context dict."""
    if not host_context:
        return []
    raw = host_context.get("actions")
    if not isinstance(raw, list):
        return []
    return [HostAction.model_validate(item) for item in raw if isinstance(item, dict)]


__all__ = [
    "ACTION_NAME_RE",
    "HOST_CONTEXT_VERSION",
    "MAX_ACTIONS",
    "MAX_ATTACHMENTS",
    "MAX_ATTACHMENT_CHARS",
    "MAX_CONTEXT_CHARS",
    "MAX_NAVIGATION",
    "Attachment",
    "HostAction",
    "HostContext",
    "HostIdentity",
    "NavigationTarget",
    "View",
    "actions_of",
    "camel_to_snake",
    "host_context_from_payload",
    "view_name_of",
]
