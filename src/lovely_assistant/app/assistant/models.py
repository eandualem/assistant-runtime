"""Request and response models for the assistant module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


# Page-scoped semantic field aliases.  Only applied inside ``active_page.data``
# for the matching page name — prevents collisions on other pages.
_PAGE_FIELD_ALIASES: dict[str, dict[str, str]] = {
    "agents": {"entities": "sessions", "coding_agents": "sessions"},
    "tasks": {"filters": "active_filters"},
}


def _apply_field_aliases(machine_state: dict[str, Any]) -> dict[str, Any]:
    """Rename semantic fields inside ``active_page.data`` based on page name."""
    active_page = machine_state.get("active_page")
    if not isinstance(active_page, dict):
        return machine_state
    page_name = active_page.get("name", "")
    aliases = _PAGE_FIELD_ALIASES.get(page_name)
    if not aliases:
        return machine_state
    data = active_page.get("data")
    if not isinstance(data, dict):
        return machine_state
    new_data = {}
    for k, v in data.items():
        new_data[aliases.get(k, k)] = v
    # Shallow copy to avoid mutating the original
    new_active_page = {**active_page, "data": new_data}
    return {**machine_state, "active_page": new_active_page}


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


class RequestConfigOverride(BaseModel):
    """Per-request config overrides sent from the dashboard."""

    model_config = ConfigDict(extra="forbid")

    default_model: str | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=100_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_turns: int | None = Field(default=None, ge=1, le=50)
    enable_working_memory: bool | None = None
    summarization_model: str | None = None
    working_memory_model: str | None = None
    default_image_model: str | None = None
    default_video_model: str | None = None
    subagent_model: str | None = None
    subagent_thinking_budget: int | None = Field(default=None, ge=1, le=100_000)


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
    parent_id: str | None
    message_type: Literal["standard", "guidance"] = Field(default="standard")
    content: str
    images: list[str] = Field(default_factory=list)
    machine_state: dict[str, Any] | None = None
    config: RequestConfigOverride | None = None
    tool_call_id: str | None = None
    tool_result: Any | None = None

    @property
    def is_continuation(self) -> bool:
        """Whether this is a continuation request (frontend returning a tool result)."""
        return self.tool_call_id is not None

    @property
    def is_guidance(self) -> bool:
        """Whether this is a mid-stream guidance message."""
        return self.message_type == "guidance"

    @property
    def message(self) -> str:
        """Compatibility accessor for internal call sites during the cutover."""
        return self.content

    @model_validator(mode="before")
    @classmethod
    def normalize_camel_case(cls, data: Any) -> Any:
        """Normalize camelCase keys from the frontend to snake_case.

        Covers three scopes:
        1. Top-level keys (sessionId → session_id, machineState → machine_state)
        2. machine_state contents (deep recursive conversion + field aliases)
        3. config keys (defaultModel → default_model, thinkingBudget → thinking_budget)
        """
        if not isinstance(data, dict):
            return data

        # 1. Normalize top-level keys
        data = {_camel_to_snake(k): v for k, v in data.items()}

        # 2. Deep-normalize machine_state contents + field aliases
        ms = data.get("machine_state")
        if ms is not None:
            normalized = _normalize_keys(ms)
            normalized = _apply_field_aliases(normalized)
            data = {**data, "machine_state": normalized}

        # 3. Normalize config keys (shallow — flat model)
        cfg = data.get("config")
        if isinstance(cfg, dict):
            data = {**data, "config": {_camel_to_snake(k): v for k, v in cfg.items()}}

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


def _build_user_prompt(message: str) -> str:
    """Build the user prompt string for the LLM.

    Images are NOT auto-attached — the agent uses the ``look_at_screen``
    tool for on-demand visual inspection instead.
    """
    return message


class AssistantResult(BaseModel):
    """Output from a single assistant interaction."""

    content: str = Field(description="Text response")
    model: str = Field(description="Model used for this turn")
    session_id: str
    turn_number: int
