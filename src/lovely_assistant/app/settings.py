"""Runtime configuration — mutable overlay on frozen AppSettings.

Three-tier config priority: per-request override > runtime overlay > frozen config default.
Resolved once at the top of each request handler via ``resolve_effective_config()``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger

from lovely_assistant.app.assistant.config import AssistantConfig

if TYPE_CHECKING:
    from lovely_assistant.services.database.interface import DatabaseService

# Sentinel to distinguish "not set" from "explicitly None"
_SENTINEL: Any = object()


@dataclass(frozen=True)
class EffectiveConfig:
    """Resolved configuration for a single request — immutable snapshot."""

    default_model: str | None
    thinking_budget: int | None
    temperature: float | None
    max_turns: int
    enable_working_memory: bool


class RuntimeSettings:
    """Mutable overlay on frozen AssistantConfig.

    Fields set to ``None`` mean "not overridden — use frozen config default".
    Thread-safe via asyncio.Lock.
    """

    _VALID_FIELDS = frozenset(
        {
            "default_model",
            "thinking_budget",
            "temperature",
            "max_turns",
            "enable_working_memory",
            "summarization_model",
            "working_memory_model",
            "default_image_model",
            "default_video_model",
            "subagent_model",
            "subagent_thinking_budget",
        }
    )

    def __init__(
        self,
        frozen_config: AssistantConfig,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._frozen = frozen_config
        self._db: DatabaseService | None = database_service
        self._lock = asyncio.Lock()
        self._updated_at: datetime | None = None

        # Overlay fields — None = not overridden
        self._default_model: str | None = None
        self._thinking_budget: int | None = None
        self._temperature: float | None = None
        self._max_turns: int | None = None
        self._enable_working_memory: bool | None = None
        self._summarization_model: str | None = None
        self._working_memory_model: str | None = None
        self._default_image_model: str | None = None
        self._default_video_model: str | None = None
        self._subagent_model: str | None = None
        self._subagent_thinking_budget: int | None = None

        # Track which fields have been explicitly set
        self._overridden: set[str] = set()

    async def update(self, **kwargs: Any) -> bool:
        """Update one or more overlay fields.

        Setting a field to ``None`` clears the override (reverts to frozen default).

        Raises:
            ValueError: If a field name is unknown or a value is out of range.
        """
        # Validate field names
        unknown = set(kwargs) - self._VALID_FIELDS
        if unknown:
            raise ValueError(f"Unknown settings field(s): {', '.join(sorted(unknown))}")

        # Validate bounds
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            t = kwargs["temperature"]
            if not (0.0 <= t <= 2.0):
                raise ValueError(f"temperature must be between 0.0 and 2.0, got {t}")

        if "thinking_budget" in kwargs and kwargs["thinking_budget"] is not None:
            tb = kwargs["thinking_budget"]
            if not (1 <= tb <= 100_000):
                raise ValueError(f"thinking_budget must be between 1 and 100000, got {tb}")

        if "subagent_thinking_budget" in kwargs and kwargs["subagent_thinking_budget"] is not None:
            stb = kwargs["subagent_thinking_budget"]
            if not (1 <= stb <= 100_000):
                raise ValueError(
                    f"subagent_thinking_budget must be between 1 and 100000, got {stb}"
                )

        if "max_turns" in kwargs and kwargs["max_turns"] is not None:
            mt = kwargs["max_turns"]
            if not (1 <= mt <= 50):
                raise ValueError(f"max_turns must be between 1 and 50, got {mt}")

        async with self._lock:
            for field, value in kwargs.items():
                setattr(self, f"_{field}", value)
                if value is None:
                    self._overridden.discard(field)
                else:
                    self._overridden.add(field)
            self._updated_at = datetime.now(UTC)

        # Persist after lock release — best-effort, don't block the caller
        return await self._persist_to_db()

    async def load_from_db(self) -> None:
        """Load persisted settings from DB into the overlay on startup.

        Non-NULL DB fields become runtime overrides. Proceeds silently if DB is unavailable.
        """
        if self._db is None:
            return

        try:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SettingsRepository,
                )

                repo = SettingsRepository(db_session)
                row = await repo.get()
                if row is None:
                    return

                for field in self._VALID_FIELDS:
                    value = getattr(row, field, None)
                    if value is not None:
                        setattr(self, f"_{field}", value)
                        self._overridden.add(field)

                if row.updated_at is not None:
                    self._updated_at = row.updated_at

            logger.info("Loaded persisted settings from DB", overrides=list(self._overridden))
        except Exception as e:
            logger.warning(
                "Failed to load settings from DB — proceeding with defaults", error=str(e)
            )

    async def _persist_to_db(self) -> bool:
        """Persist current overlay state to DB. Best-effort — failures are logged, not raised."""
        if self._db is None:
            return False

        # Build full state dict: overridden fields get values, others get None (clears old values)
        state: dict[str, Any] = {}
        for field in self._VALID_FIELDS:
            if field in self._overridden:
                state[field] = getattr(self, f"_{field}")
            else:
                state[field] = None

        try:
            async with self._db.session_context() as db_session:
                from lovely_assistant.services.database.repositories import (
                    SettingsRepository,
                )

                repo = SettingsRepository(db_session)
                await repo.save(state)
            return True
        except Exception as e:
            logger.warning("Failed to persist settings to DB", error=str(e))
            return False

    def get(self, field: str, frozen_default: Any = None) -> Any:
        """Return the runtime override if set, otherwise frozen_default.

        Simple consumer API — returns just the value (no source annotation).
        Used by services that need runtime > frozen two-tier resolution.
        """
        if field in self._overridden:
            return getattr(self, f"_{field}")
        return frozen_default

    def _resolve_field(self, field: str) -> tuple[Any, str]:
        """Return (value, source) for a field."""
        overlay = getattr(self, f"_{field}")
        if field in self._overridden:
            return overlay, "runtime"
        frozen_val = getattr(self._frozen, field, None)
        return frozen_val, "config_default"

    def to_response_dict(self) -> dict[str, Any]:
        """Serialize current state for the GET /settings response."""
        values: dict[str, dict[str, Any]] = {}
        for field in self._VALID_FIELDS:
            value, source = self._resolve_field(field)
            values[field] = {"value": value, "source": source}

        return {
            "values": values,
            "updated_at": self._updated_at.isoformat() if self._updated_at else None,
        }


def resolve_effective_config(
    frozen_config: AssistantConfig,
    runtime_settings: RuntimeSettings | None = None,
    per_request: Any | None = None,
) -> EffectiveConfig:
    """Resolve three-tier config priority into a frozen snapshot.

    Priority: per_request > runtime_settings > frozen_config.

    Args:
        frozen_config: The frozen AssistantConfig from env/startup.
        runtime_settings: Optional mutable overlay from PATCH /settings.
        per_request: Optional RequestConfigOverride from the chat request body.
    """

    def _pick(field: str, frozen_default: Any) -> Any:
        """First non-sentinel value wins: per_request → runtime → frozen."""
        # Check per-request override
        if per_request is not None:
            val = getattr(per_request, field, None)
            if val is not None:
                return val

        # Check runtime overlay
        if runtime_settings is not None and field in runtime_settings._overridden:
            return getattr(runtime_settings, f"_{field}")

        # Fall back to frozen config
        return frozen_default

    return EffectiveConfig(
        default_model=_pick("default_model", frozen_config.default_model),
        thinking_budget=_pick("thinking_budget", frozen_config.thinking_budget),
        temperature=_pick("temperature", frozen_config.temperature),
        max_turns=_pick("max_turns", frozen_config.max_turns),
        enable_working_memory=_pick("enable_working_memory", frozen_config.enable_working_memory),
    )
