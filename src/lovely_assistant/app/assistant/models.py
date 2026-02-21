"""Request and response models for the assistant module."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.messages import BinaryContent, ImageUrl, UserContent

from lovely_assistant.services.tools.models import DeferredToolRequest

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


class RequestConfigOverride(BaseModel):
    """Per-request config overrides sent from the dashboard."""

    model_config = ConfigDict(extra="forbid")

    default_model: str | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=100_000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_turns: int | None = Field(default=None, ge=1, le=50)
    enable_working_memory: bool | None = None


class AssistantRequest(BaseModel):
    """Input for a single assistant interaction."""

    session_id: str
    message: str
    images: list[str] = Field(default_factory=list)
    machine_state: dict[str, Any] | None = None
    config: RequestConfigOverride | None = None
    # For continuations (frontend returning tool result):
    tool_call_id: str | None = None
    tool_result: Any = None

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


def _build_user_prompt(
    message: str,
    images: list[str],
    model: str | None = None,
) -> str | list[UserContent]:
    """Convert message + images into a Pydantic AI user prompt.

    Returns plain ``str`` when no images (zero behavioral change to existing path).
    Returns ``list[UserContent]`` when valid images are present.
    """
    if not images:
        return message

    # Vision capability warning (advisory only — don't block)
    if model is not None:
        _warn_if_no_vision(model)

    converted: list[UserContent] = []
    for img in images:
        try:
            if img.startswith("data:"):
                converted.append(BinaryContent.from_data_uri(img))
            elif img.startswith("https://"):
                converted.append(ImageUrl(url=img))
            else:
                logger.warning("Skipping image with unsupported scheme", image_prefix=img[:30])
        except Exception as e:
            logger.warning("Skipping malformed image", error=str(e))

    if not converted:
        return message

    return [message, *converted]


def _warn_if_no_vision(model: str) -> None:
    """Log a warning if the model is known to lack vision capability."""
    from lovely_assistant.app.models_registry import MODEL_CATALOG

    for entry in MODEL_CATALOG:
        if entry.id == model:
            if "vision" not in entry.capabilities:
                logger.warning(
                    "Model may not support vision",
                    model=model,
                    capabilities=entry.capabilities,
                )
            return
    # Model not in catalog — don't warn (registry isn't exhaustive)


class AssistantResult(BaseModel):
    """Output from a single assistant interaction."""

    content: str | None = Field(
        default=None, description="Text response (None if deferred tool call)"
    )
    model: str = Field(description="Model used for this turn")
    deferred_tool_request: DeferredToolRequest | None = Field(
        default=None, description="Frontend tool call to execute"
    )
    session_id: str
    turn_number: int

    @property
    def is_tool_call(self) -> bool:
        """Whether this result is a deferred tool call (not text)."""
        return self.deferred_tool_request is not None
