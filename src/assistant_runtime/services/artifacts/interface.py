"""ArtifactService — the evolving prompt artifacts of one assistant profile.

Public facade for the artifacts module. Implements LifecycleAware.

Every mutation goes through this class, which applies the profile's
:class:`ArtifactPolicy` from code: the artifact text itself can never
change what the assistant may do to it. With a reachable database the
versions are durable; otherwise they live in memory for the process and
every result says so (``durable=False``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from assistant_runtime.artifacts import ArtifactDefinition, AssistantProfile
from assistant_runtime.services.artifacts._store import (
    ArtifactStore,
    DatabaseArtifactStore,
    InMemoryArtifactStore,
)
from assistant_runtime.services.artifacts.config import ArtifactsConfig
from assistant_runtime.services.artifacts.exceptions import (
    ArtifactConflictError,
    ArtifactError,
    ArtifactPermissionError,
    ArtifactVersionNotFoundError,
    UnknownArtifactError,
)
from assistant_runtime.services.artifacts.models import Actor, ArtifactVersion, MutationResult

if TYPE_CHECKING:
    from assistant_runtime.services.database.interface import DatabaseService

Action = Literal["propose", "update", "activate", "delete"]


class ArtifactService:
    """Read active texts for the prompt; write versions under the profile's policies."""

    def __init__(
        self,
        config: ArtifactsConfig,
        profile: AssistantProfile,
        database_service: DatabaseService | None = None,
    ) -> None:
        self._config = config
        self._profile = profile
        self._database_service = database_service
        self._store: ArtifactStore | None = None
        self._cache: dict[str, str] | None = None
        self._cached_at = 0.0
        self._started = False

    async def start(self) -> None:
        """Choose the store: Postgres when reachable, else process memory."""
        database = self._database_service
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseArtifactStore(database)
        else:
            self._store = InMemoryArtifactStore()
        self._cache = None
        self._started = True
        logger.info(
            "Artifact service started",
            profile=self._profile.name,
            artifacts=list(self._profile.names),
            durable=self._store.durable,
        )

    async def stop(self) -> None:
        self._store = None
        self._cache = None
        self._started = False
        logger.info("Artifact service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._started,
            "profile": self._profile.name,
            "artifacts": len(self._profile.artifacts),
            "durable": self._store.durable if self._store is not None else None,
        }

    @property
    def profile(self) -> AssistantProfile:
        return self._profile

    @property
    def durable(self) -> bool:
        """Whether versions survive a process restart."""
        return self._require_store().durable

    # --- Reads ---

    async def active_texts(self) -> dict[str, str]:
        """Text per artifact for the prompt: the active version, else the default.

        Cached for ``cache_ttl_seconds``; a mutation through this service
        invalidates the cache, so it affects the next prompt built.
        """
        store = self._require_store()
        now = time.monotonic()
        if self._cache is not None and now - self._cached_at < self._config.cache_ttl_seconds:
            return dict(self._cache)
        texts = self._profile.defaults
        try:
            for row in await store.get_all_active(self._profile.name):
                if row.name in texts:
                    texts[row.name] = row.content
        except Exception as e:
            logger.warning("Failed to load artifacts, using defaults", error=str(e))
            return texts
        self._cache = dict(texts)
        self._cached_at = now
        return texts

    def invalidate(self) -> None:
        """Forget cached texts so the next prompt reads the store."""
        self._cache = None

    async def list_active(self) -> list[ArtifactVersion]:
        """Active versions of the profile's artifacts, in prompt order."""
        rows = await self._require_store().get_all_active(self._profile.name)
        known = [row for row in rows if self._profile.get(row.name) is not None]
        return sorted(known, key=lambda row: self._profile.sort_key(row.name))

    async def get_active(self, name: str) -> ArtifactVersion | None:
        self._definition(name)
        return await self._require_store().get_active(self._profile.name, name)

    async def history(self, name: str, limit: int | None = None) -> list[ArtifactVersion]:
        self._definition(name)
        return await self._require_store().get_history(
            self._profile.name, name, limit or self._config.history_limit
        )

    def allowed_actions(self, name: str, actor_kind: Literal["assistant", "host"]) -> list[str]:
        """The mutations ``actor_kind`` may perform on ``name``."""
        artifact = self._definition(name)
        return [
            action
            for action in ("propose", "update", "activate", "delete")
            if self._permitted(artifact, Actor(kind=actor_kind), action)
        ]

    # --- Writes ---

    async def propose(
        self, name: str, content: str, *, actor: Actor, expected_version: int | None = None
    ) -> MutationResult:
        """Store a new inactive version; an authorized actor activates it later."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "propose")
        content = self._clean_content(name, content)
        active = await self._check_expected(name, expected_version)
        if active is not None and active.content == content:
            return self._result(active, activated=True, live=active, unchanged=True)
        row = await self._require_store().propose(
            self._profile.name, name, content, actor.proposed_by
        )
        logger.info("Proposed artifact version", name=name, version=row.version, by=actor.kind)
        return self._result(row, activated=False, live=active)

    async def update(
        self, name: str, content: str, *, actor: Actor, expected_version: int | None = None
    ) -> MutationResult:
        """Store a new version and activate it at once (autonomous edit)."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "update")
        content = self._clean_content(name, content)
        active = await self._check_expected(name, expected_version)
        if active is not None and active.content == content:
            return self._result(active, activated=True, live=active, unchanged=True)
        store = self._require_store()
        row = await store.propose(self._profile.name, name, content, actor.proposed_by)
        activated = await store.activate(self._profile.name, name, row.version)
        self.invalidate()
        logger.info("Updated artifact", name=name, version=row.version, by=actor.kind)
        return self._result(activated or row, activated=True, live=activated or row)

    async def activate(self, name: str, version: int, *, actor: Actor) -> MutationResult:
        """Make ``version`` the active one (approval or rollback)."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "activate")
        row = await self._require_store().activate(self._profile.name, name, version)
        if row is None:
            raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
        self.invalidate()
        logger.info("Activated artifact version", name=name, version=version, by=actor.kind)
        return self._result(row, activated=True, live=row)

    async def delete(self, name: str, *, actor: Actor) -> int:
        """Remove every stored version; the default text applies again."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "delete")
        count = await self._require_store().delete(self._profile.name, name)
        self.invalidate()
        logger.info("Deleted artifact versions", name=name, count=count, by=actor.kind)
        return count

    # --- Internals ---

    def _require_store(self) -> ArtifactStore:
        if self._store is None:
            raise ArtifactError("Artifact service not started")
        return self._store

    def _definition(self, name: str) -> ArtifactDefinition:
        artifact = self._profile.get(name)
        if artifact is None:
            raise UnknownArtifactError(
                f"Unknown artifact '{name}'. Known artifacts: {self._profile.names_text()}."
            )
        return artifact

    @staticmethod
    def _permitted(artifact: ArtifactDefinition, actor: Actor, action: str) -> bool:
        policy = artifact.policy
        if actor.kind == "host":
            return policy.host_edit
        if action == "propose":
            return policy.assistant_edit in ("propose", "autonomous")
        if action == "update":
            return policy.assistant_edit == "autonomous"
        if action == "activate":
            return policy.assistant_activate
        return False

    def _authorize(self, artifact: ArtifactDefinition, actor: Actor, action: Action) -> None:
        if not self._permitted(artifact, actor, action):
            raise ArtifactPermissionError(
                f"The {actor.kind} may not {action} artifact '{artifact.name}'"
            )

    @staticmethod
    def _clean_content(name: str, content: str) -> str:
        content = (content or "").strip()
        if not content:
            raise ArtifactError(f"Content is required to write artifact '{name}'")
        return content

    async def _check_expected(
        self, name: str, expected_version: int | None
    ) -> ArtifactVersion | None:
        active = await self._require_store().get_active(self._profile.name, name)
        if expected_version is not None:
            current = active.version if active is not None else 0
            if current != expected_version:
                raise ArtifactConflictError(
                    f"Artifact '{name}' is at version {current}, not {expected_version}"
                )
        return active

    def _result(
        self,
        row: ArtifactVersion,
        *,
        activated: bool,
        live: ArtifactVersion | None,
        unchanged: bool = False,
    ) -> MutationResult:
        return MutationResult(
            version=row,
            activated=activated,
            live_version=live.version if live is not None else None,
            unchanged=unchanged,
            durable=self._require_store().durable,
        )
