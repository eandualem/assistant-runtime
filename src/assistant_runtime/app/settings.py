"""Runtime configuration — the mutable overlay between frozen settings and the request.

Three tiers, most specific wins: the request's ``config`` > the runtime
overlay (``PATCH /api/settings``) > the frozen ``AssistantConfig``.
``resolve_effective_config()`` folds them once per request. The tunables
themselves are declared once, in ``TunableOverrides``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from assistant_runtime.app.assistant.config import (
    TUNABLE_FIELDS,
    AssistantConfig,
    TunableOverrides,
)

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService


@dataclass(frozen=True)
class EffectiveConfig:
    """The resolved configuration for one request.

    One attribute per tunable (``tests/unit/app/test_settings.py`` checks the
    two stay in step); the fields the frozen config always provides are typed
    accordingly.
    """

    default_model: str | None
    thinking_budget: int | None
    temperature: float | None
    max_turns: int
    enable_working_memory: bool
    summarization_model: str | None
    working_memory_model: str | None
    default_image_model: str | None
    default_video_model: str | None
    subagent_model: str | None
    subagent_thinking_budget: int | None


assert {f.name for f in dataclasses.fields(EffectiveConfig)} == TUNABLE_FIELDS


class RuntimeSettings:
    """Mutable overlay on the frozen ``AssistantConfig``.

    Holds only the tunables that were explicitly set; everything else falls
    through to the frozen default. Persisted to Postgres when it is reachable.
    """

    def __init__(
        self,
        frozen_config: AssistantConfig,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._frozen = frozen_config
        self._db: DatabaseService | None = database_service
        self._lock = asyncio.Lock()
        self._updated_at: datetime | None = None
        self._overrides: dict[str, Any] = {}

    @property
    def overrides(self) -> dict[str, Any]:
        """The tunables currently overridden, by name."""
        return dict(self._overrides)

    async def update(self, **kwargs: Any) -> bool:
        """Set or clear overrides; a ``None`` value clears one.

        Returns whether the new state was persisted.

        Raises:
            ValueError: An unknown field or a value out of range.
        """
        unknown = set(kwargs) - TUNABLE_FIELDS
        if unknown:
            raise ValueError(f"Unknown settings field(s): {', '.join(sorted(unknown))}")
        try:
            validated = TunableOverrides(**kwargs)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            raise ValueError(f"Invalid settings: {problems}") from e

        async with self._lock:
            for field in kwargs:
                value = getattr(validated, field)
                if value is None:
                    self._overrides.pop(field, None)
                else:
                    self._overrides[field] = value
            self._updated_at = datetime.now(UTC)

        return await self._persist_to_db()

    async def load_from_db(self) -> None:
        """Load persisted overrides on startup; silently skipped without a database.

        The database may have come up after the service probed it at startup,
        so an unhealthy service is probed once more before giving up.
        """
        if self._db is None:
            return
        if not self._db.healthy:
            await self._db.health_check()
            if not self._db.healthy:
                return
        try:
            from assistant_runtime.services.database.repositories import SettingsRepository

            async with self._db.session_context() as db_session:
                row = await SettingsRepository(db_session).get()
                if row is None:
                    return
                for field in TUNABLE_FIELDS:
                    value = getattr(row, field, None)
                    if value is not None:
                        self._overrides[field] = value
                if row.updated_at is not None:
                    self._updated_at = row.updated_at
            logger.info("Loaded persisted settings from DB", overrides=sorted(self._overrides))
        except Exception as e:
            logger.warning(
                "Failed to load settings from DB — proceeding with defaults", error=str(e)
            )

    async def _persist_to_db(self) -> bool:
        """Write the full overlay (unset fields as NULL). Best-effort."""
        if self._db is None or not getattr(self._db, "healthy", True):
            return False
        state = {field: self._overrides.get(field) for field in TUNABLE_FIELDS}
        try:
            from assistant_runtime.services.database.repositories import SettingsRepository

            async with self._db.session_context() as db_session:
                await SettingsRepository(db_session).save(state)
            return True
        except Exception as e:
            logger.warning("Failed to persist settings to DB", error=str(e))
            return False

    def get(self, field: str, frozen_default: Any = None) -> Any:
        """The runtime override for ``field``, else ``frozen_default``."""
        return self._overrides.get(field, frozen_default)

    def to_response_dict(self) -> dict[str, Any]:
        """Every tunable with its value and which tier it came from (``GET /settings``)."""
        values = {}
        for field in sorted(TUNABLE_FIELDS):
            if field in self._overrides:
                values[field] = {"value": self._overrides[field], "source": "runtime"}
            else:
                values[field] = {
                    "value": getattr(self._frozen, field, None),
                    "source": "config_default",
                }
        return {
            "values": values,
            "updated_at": self._updated_at.isoformat() if self._updated_at else None,
        }


# Cost-bearing tunables an untrusted request may lower but never raise past
# the host's (runtime or frozen) value.
CEILING_FIELDS: frozenset[str] = frozenset(
    {"max_turns", "thinking_budget", "subagent_thinking_budget"}
)


def resolve_effective_config(
    frozen_config: AssistantConfig,
    runtime_settings: RuntimeSettings | None = None,
    per_request: TunableOverrides | None = None,
) -> EffectiveConfig:
    """Fold the three tiers into one frozen snapshot: request > runtime > frozen.

    The runtime overlay is administration and trusted. The request body is
    not: for ``CEILING_FIELDS`` it can only narrow the host's value.
    """
    request_values = per_request.model_dump(exclude_none=True) if per_request else {}
    runtime_values = runtime_settings.overrides if runtime_settings else {}

    def _trusted(field: str) -> Any:
        if field in runtime_values:
            return runtime_values[field]
        return getattr(frozen_config, field, None)

    def _pick(field: str) -> Any:
        if field in request_values:
            requested = request_values[field]
            ceiling = _trusted(field)
            if field in CEILING_FIELDS and ceiling is not None and requested > ceiling:
                logger.debug("Request override clamped", field=field, requested=requested)
                return ceiling
            return requested
        return _trusted(field)

    return EffectiveConfig(**{field: _pick(field) for field in TUNABLE_FIELDS})
