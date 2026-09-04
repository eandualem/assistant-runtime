"""Request and response models for the assistant module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from assistant_runtime.app.assistant.config import TunableOverrides

# --- camelCase → snake_case normalization ---

_CAMEL_RE_1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    """Convert a camelCase or PascalCase string to snake_case.

    Handles consecutive capitals: ``eventType`` → ``event_type``,
    ``HTMLParser`` → ``html_parser``, already-snake → unchanged.
    """
    s = _CAMEL_RE_1.sub(r"\1_\2", name)
    return _CAMEL_RE_2.sub(r"\1_\2", s).lower()


def _normalize_keys(obj: Any) -> Any:
    """Recursively convert all dict keys from camelCase to snake_case."""
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


# Older clients sent the host context under other names. They are accepted and
# mapped onto the documented shape so a host can migrate at its own pace.
_LEGACY_TOP_LEVEL_KEYS: dict[str, str] = {"machine_state": "host_context"}
_LEGACY_CONTEXT_KEYS: dict[str, str] = {"active_page": "page"}
_LEGACY_PAGE_KEYS: dict[str, str] = {"machines": "state", "available_actions": "actions"}


def _rename_keys(obj: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    return {aliases.get(k, k): v for k, v in obj.items()}


def _apply_context_aliases(host_context: dict[str, Any]) -> dict[str, Any]:
    """Map legacy key names inside a host context onto the documented shape."""
    context = _rename_keys(host_context, _LEGACY_CONTEXT_KEYS)
    page = context.get("page")
    if isinstance(page, dict):
        context = {**context, "page": _rename_keys(page, _LEGACY_PAGE_KEYS)}
    return context


def normalize_host_context(raw: Any) -> dict[str, Any] | None:
    """A host context as the runtime expects it: snake_case keys, documented shape.

    Accepts what a client sent (camelCase or snake_case, current or legacy key
    names) and returns None for anything that is not a mapping. Every entry
    point that takes a host context (the request model, the Socket.IO join)
    goes through here so the stored ``last_host_context`` is always canonical.
    """
    if not isinstance(raw, dict):
        return None
    return _apply_context_aliases(_normalize_keys(raw))


_HOST_CONTEXT_PAYLOAD_KEYS = ("host_context", "hostContext", "machine_state", "machineState")


def host_context_from_payload(data: Any) -> dict[str, Any] | None:
    """Pull and normalise the host context out of a raw client payload, if any."""
    if not isinstance(data, dict):
        return None
    for key in _HOST_CONTEXT_PAYLOAD_KEYS:
        if key in data:
            return normalize_host_context(data[key])
    return None


@dataclass(frozen=True)
class PromptResult:
    """Result of building a system prompt — content plus fragment metadata."""

    content: str
    fragments: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentSetupContext:
    """Everything needed to run an agent — produced by prepare_agent_context().

    Shared by both AssistantService and StreamingService to prevent drift.
    """

    agent: Any  # pydantic_ai.Agent
    available_tools: Any  # ToolSet
    toolsets: list[Any]
    prompt_result: PromptResult
    resolved_model: str
    usage_limits: Any  # pydantic_ai.usage.UsageLimits
    output_type: Any  # str
    effective_config: Any  # EffectiveConfig
    mcp_summary: list[dict[str, Any]] | None


# Top-level keys that may carry screenshot data from the frontend.
# Includes both camelCase and snake_case variants since Pydantic v2 runs
# mode="before" validators in reverse definition order (this validator
# may execute before normalize_camel_case).
_SCREENSHOT_TOP_LEVEL_KEYS = (
    "screenshot",
    "image",
    "image_data_uri",
    "imageDataUri",
    "data_uri",
    "dataUri",
)


class AssistantRequest(BaseModel):
    """Input for a single assistant interaction."""

    id: str
    session_id: str
    parent_id: str | None = None
    message_type: Literal["standard", "steering"] = Field(default="standard")
    content: str
    images: list[str] = Field(default_factory=list)
    host_context: dict[str, Any] | None = None
    config: TunableOverrides | None = None
    tool_call_id: str | None = None
    tool_result: Any | None = None

    @property
    def is_continuation(self) -> bool:
        """Whether this is a continuation request (frontend returning a tool result)."""
        return self.tool_call_id is not None

    @property
    def is_steering(self) -> bool:
        """Whether this is a mid-stream steering message."""
        return self.message_type == "steering"

    @property
    def message(self) -> str:
        """Compatibility accessor for internal call sites during the cutover."""
        return self.content

    @model_validator(mode="after")
    def validate_message_shape(self) -> AssistantRequest:
        """Enforce distinct wire contracts for standard messages vs steering."""
        if self.is_steering:
            if self.parent_id is not None:
                raise ValueError("Steering requests must not include parent_id")
            if self.tool_call_id is not None or self.tool_result is not None:
                raise ValueError("Steering requests must not include tool continuation fields")
            return self

        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_camel_case(cls, data: Any) -> Any:
        """Normalize camelCase keys from the frontend to snake_case.

        Covers three scopes:
        1. Top-level keys (sessionId → session_id, hostContext → host_context),
           plus the legacy ``machine_state`` name for the host context
        2. host_context contents (deep recursive conversion + legacy key aliases)
        3. config keys (defaultModel → default_model, thinkingBudget → thinking_budget)
        """
        if not isinstance(data, dict):
            return data

        # 1. Normalize top-level keys
        data = _rename_keys(
            {_camel_to_snake(k): v for k, v in data.items()}, _LEGACY_TOP_LEVEL_KEYS
        )

        # 2. Deep-normalize host_context contents + legacy aliases
        ctx = data.get("host_context")
        if isinstance(ctx, dict):
            data = {**data, "host_context": normalize_host_context(ctx)}

        # 3. Normalize config keys (shallow — flat model)
        cfg = data.get("config")
        if isinstance(cfg, dict):
            data = {**data, "config": {_camel_to_snake(k): v for k, v in cfg.items()}}

        if data.get("message_type") == "guidance":
            data = {**data, "message_type": "steering"}

        return data

    @model_validator(mode="before")
    @classmethod
    def fold_screenshot_into_images(cls, data: Any) -> Any:
        """Capture top-level screenshot fields and fold them into images[].

        If a recognized top-level key holds a data URI string, prepend it
        to images[] so extract_screenshot_data_uri() can find it.
        Checks both camelCase and snake_case variants since validator
        ordering with normalize_camel_case is not guaranteed.
        """
        if not isinstance(data, dict):
            return data

        for key in _SCREENSHOT_TOP_LEVEL_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.startswith("data:image/"):
                existing = data.get("images") or []
                if not isinstance(existing, list):
                    existing = [existing]
                data = {**data, "images": [value, *existing]}
                break  # first match wins

        return data


class AssistantResult(BaseModel):
    """The final answer of one turn (the non-streaming form of ``final_response``)."""

    content: str = Field(description="Text response")
    model: str = Field(description="Model used for this turn")
    session_id: str
    turn_number: int
    message_id: str | None = Field(default=None, description="Id of the assistant message row")
    pending_tool_call: dict[str, Any] | None = Field(
        default=None,
        description="Set when the turn ended on a host tool call the client must answer",
    )
