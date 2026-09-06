"""Request and response models for the assistant module."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.host_context import (
    Attachment,
    HostContext,
    camel_to_snake,
)

# --- camelCase → snake_case normalization ---

_camel_to_snake = camel_to_snake


def _dedupe_attachments(items: list[Any]) -> list[Any]:
    """Keep the first of attachments that carry the same content for the same purpose."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[Any] = []
    for item in items:
        if isinstance(item, dict):
            key = (
                item.get("purpose", "reference"),
                item.get("data_uri"),
                item.get("url"),
                item.get("text"),
            )
            if key in seen:
                continue
            seen.add(key)
        unique.append(item)
    return unique


def normalize_host_context(raw: Any) -> dict[str, Any] | None:
    """A host context in its canonical form, or None for anything that is not a mapping.

    Every entry point that takes a host context (the request model, the
    Socket.IO join) goes through ``HostContext.from_payload``, so the stored
    ``last_host_context`` is always the validated, snake_case, versioned form.
    Invalid content raises ``ValueError``.
    """
    context = HostContext.from_payload(raw)
    return context.to_dict() if context is not None else None


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
    deps: Any = None


# Keys under which a host may carry a screenshot data URI, at the top level
# of the request or inside a host tool's result. Both camelCase and
# snake_case: the screenshot validator may run before key normalisation.
SCREENSHOT_KEYS = (
    "screenshot",
    "image",
    "image_data_uri",
    "imageDataUri",
    "data_uri",
    "dataUri",
)


def _find_screenshot_data_uri(value: Any) -> str | None:
    """Recursively search common payload shapes for an image data URI."""
    if isinstance(value, str):
        return value if value.startswith("data:image/") else None
    if isinstance(value, dict):
        for key in SCREENSHOT_KEYS:
            if key in value:
                found = _find_screenshot_data_uri(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = _find_screenshot_data_uri(child)
            if found is not None:
                return found
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found = _find_screenshot_data_uri(item)
            if found is not None:
                return found
    return None


def extract_screenshot_data_uri(
    *,
    images: Sequence[str] | None = None,
    tool_result: Any | None = None,
) -> str | None:
    """The freshest screenshot: the request's images first, then the tool result."""
    if images:
        for image in images:
            if isinstance(image, str) and image:
                return image
    return _find_screenshot_data_uri(tool_result)


def strip_screenshot_from_tool_result(tool_result: Any) -> Any:
    """A copy of a host tool result with every image data URI replaced by a placeholder.

    Mirrors ``extract_screenshot_data_uri``, which finds a screenshot under any
    key: whatever it can find, this removes, so the base64 blob never reaches
    the model as tool output (it goes to the request context for
    ``look_at_screen`` instead).
    """
    if isinstance(tool_result, str):
        if tool_result.startswith("data:image/"):
            return "[screenshot captured — use look_at_screen to inspect]"
        return tool_result
    if isinstance(tool_result, list):
        return [strip_screenshot_from_tool_result(item) for item in tool_result]
    if isinstance(tool_result, dict):
        return {key: strip_screenshot_from_tool_result(value) for key, value in tool_result.items()}
    return tool_result


class AssistantRequest(BaseModel):
    """Input for a single assistant interaction."""

    id: str
    session_id: str
    parent_id: str | None = None
    message_type: Literal["standard", "steering"] = Field(default="standard")
    content: str
    images: list[str] = Field(default_factory=list)
    """Legacy: data URIs treated as screenshots. Prefer ``attachments``."""
    attachments: list[Attachment] = Field(default_factory=list)
    host_context: dict[str, Any] | None = None
    """The canonical form of a ``HostContext`` (see ``host_context.py``)."""
    config: TunableOverrides | None = None
    tool_call_id: str | None = None
    tool_result: Any | None = None
    tool_outcome: Literal["success", "failed"] = "success"
    """Continuation only: whether the host completed the action or it failed."""

    @property
    def screenshot(self) -> str | None:
        """The data URI of the screenshot attachment, if the host sent one."""
        for attachment in self.attachments:
            if attachment.purpose == "screenshot" and attachment.data_uri:
                return attachment.data_uri
        return None

    @property
    def reference_attachments(self) -> list[Attachment]:
        """Attachments that go into the model's message as native content."""
        return [a for a in self.attachments if a.purpose == "reference"]

    @property
    def is_continuation(self) -> bool:
        """Whether this is a continuation request (host returning a tool result)."""
        return self.tool_call_id is not None

    @property
    def is_steering(self) -> bool:
        """Whether this is a mid-stream steering message."""
        return self.message_type == "steering"

    @model_validator(mode="after")
    def validate_message_shape(self) -> AssistantRequest:
        """Enforce distinct wire contracts for standard messages vs steering."""
        if self.is_steering:
            if self.parent_id is not None:
                raise ValueError("Steering requests must not include parent_id")
            if self.tool_call_id is not None or self.tool_result is not None:
                raise ValueError("Steering requests must not include tool continuation fields")
            return self

        if self.tool_outcome != "success" and self.tool_call_id is None:
            raise ValueError("tool_outcome requires tool_call_id (a continuation)")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_camel_case(cls, data: Any) -> Any:
        """Normalize camelCase keys from the host to snake_case.

        Covers three scopes:
        1. Top-level keys (sessionId → session_id, hostContext → host_context)
        2. host_context contents (deep recursive conversion)
        3. config keys (defaultModel → default_model, thinkingBudget → thinking_budget)
        """
        if not isinstance(data, dict):
            return data

        # 1. Normalize top-level keys
        data = {_camel_to_snake(k): v for k, v in data.items()}

        # 2. Validate host_context into its canonical form (aliases, camelCase)
        ctx = data.get("host_context")
        if isinstance(ctx, dict):
            data = {**data, "host_context": normalize_host_context(ctx)}

        # 2b. Legacy screenshot fields and images[] become screenshot attachments
        images = list(data.get("images") or []) if isinstance(data.get("images"), list) else []
        for key in ("screenshot", "image", "image_data_uri", "data_uri"):
            value = data.pop(key, None)
            if isinstance(value, str) and value.startswith("data:image/"):
                images.append(value)
        attachments = [
            {camel_to_snake(k): v for k, v in item.items()} if isinstance(item, dict) else item
            for item in (data.get("attachments") or [])
        ]
        for image in images:
            if isinstance(image, str) and image:
                attachments.append({"kind": "image", "purpose": "screenshot", "data_uri": image})
        # Attachments carried inside host_context reach the turn the same way;
        # message-level ones come first and duplicates are dropped.
        context = data.get("host_context")
        if isinstance(context, dict):
            attachments.extend(
                item for item in context.get("attachments", []) if isinstance(item, dict)
            )
        data = {**data, "images": images, "attachments": _dedupe_attachments(attachments)}

        # 3. Normalize config keys (shallow — flat model)
        cfg = data.get("config")
        if isinstance(cfg, dict):
            data = {**data, "config": {_camel_to_snake(k): v for k, v in cfg.items()}}

        return data

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
