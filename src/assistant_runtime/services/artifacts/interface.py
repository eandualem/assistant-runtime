"""ArtifactService — registered profiles and their evolving prompt artifacts.

Public facade for the artifacts module. Implements LifecycleAware.

Every mutation goes through this class, which applies the profile's
:class:`ArtifactPolicy` from code: the artifact text itself can never
change what the assistant may do to it. With a reachable database the
versions are durable; otherwise they live in memory for the process and
every result says so (``durable=False``).
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Sequence
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
    UnknownProfileError,
)
from assistant_runtime.services.artifacts.models import Actor, ArtifactVersion, MutationResult

if TYPE_CHECKING:
    from assistant_runtime.base.events import EventHub
    from assistant_runtime.services.database.interface import DatabaseService

Action = Literal["propose", "update", "activate", "reject", "delete"]
SEED = "seed"


class ArtifactService:
    """Read active texts for the prompt; write versions under the profile's policies."""

    def __init__(
        self,
        config: ArtifactsConfig,
        profile: AssistantProfile,
        database_service: DatabaseService | None = None,
        *,
        profiles: Sequence[AssistantProfile] = (),
        events: EventHub | None = None,
    ) -> None:
        self._config = config
        self._profile = profile
        self._database_service = database_service
        self._events = events
        self._store: ArtifactStore | None = None
        self._cache: dict[str, str] | None = None
        self._cached_at = 0.0
        self._cache_generation = 0
        self._started = False
        self._owner = self
        self._views: dict[str, ArtifactService] = {profile.name: self}
        for additional in profiles:
            if additional.name in self._views:
                raise ValueError(f"Duplicate assistant profile name: {additional.name!r}")
            view = ArtifactService(config, additional, database_service)
            view._owner = self
            view._views = self._views
            self._views[additional.name] = view

    @property
    def available_profiles(self) -> tuple[str, ...]:
        """Registered profile names, with the default first."""
        return tuple(self._views)

    def for_profile(self, name: str | None) -> ArtifactService:
        """Return the stable scoped view; omitted names select the startup default."""
        if name is None:
            return self._owner
        try:
            return self._views[name]
        except KeyError:
            raise UnknownProfileError(
                f"Unknown assistant profile {name!r}. "
                f"Available profiles: {', '.join(self.available_profiles)}."
            ) from None

    async def start(self) -> None:
        """Choose the store: Postgres when reachable, else process memory."""
        if self is not self._owner:
            await self._owner.start()
            return
        if self._started:
            return
        database = self._database_service
        if database is not None and getattr(database, "healthy", False):
            self._store = DatabaseArtifactStore(database)
        else:
            self._store = InMemoryArtifactStore()
        if self._store.durable and getattr(database, "required", False) is True:
            await self._seed_defaults()
        for view in self._views.values():
            view.invalidate()
        self._started = True
        logger.info(
            "Artifact service started",
            profile=self._profile.name,
            artifacts=list(self._profile.names),
            durable=self._store.durable,
        )

    async def _seed_defaults(self) -> None:
        """Store each default as version 1 where nothing is stored yet.

        With ``DATABASE__REQUIRED`` the active text is then always a
        versioned record; the definition's default is only its seed.
        """
        assert self._store is not None
        for view in self._views.values():
            scope = view._profile.name
            for artifact in view._profile.artifacts:
                content = (artifact.default or "").strip()
                if not content:
                    continue
                async with self._store.transaction(scope, artifact.name) as store:
                    if await store.get_history(scope, artifact.name, 1):
                        continue
                    row = await store.propose(scope, artifact.name, content, SEED, actor_kind=SEED)
                    await store.activate(scope, artifact.name, row.version)
                logger.info("Seeded artifact from its default", profile=scope, name=artifact.name)

    async def stop(self) -> None:
        if self is not self._owner:
            await self._owner.stop()
            return
        self._store = None
        for view in self._views.values():
            view.invalidate()
        self._started = False
        logger.info("Artifact service stopped")

    async def health_check(self) -> dict[str, Any]:
        return {
            "healthy": self._owner._started,
            "profile": self._profile.name,
            "artifacts": len(self._profile.artifacts),
            "durable": self._owner._store.durable if self._owner._store is not None else None,
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
        generation = self._cache_generation
        texts = self._profile.defaults
        try:
            for row in await store.get_all_active(self._profile.name):
                if row.name in texts:
                    texts[row.name] = row.content
        except Exception as e:
            # With DATABASE__REQUIRED the stored versions are the only
            # source; a prompt from defaults would silently drop them.
            if getattr(self._database_service, "required", False):
                raise ArtifactError(f"Failed to load artifacts: {e}") from e
            logger.warning("Failed to load artifacts, using defaults", error=str(e))
            return texts
        if generation == self._cache_generation:
            self._cache = dict(texts)
            self._cached_at = now
        return texts

    def invalidate(self) -> None:
        """Forget cached texts so the next prompt reads the store."""
        self._cache = None
        self._cache_generation += 1

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

    async def proposals(self, status: str = "pending") -> list[dict[str, Any]]:
        """The profile's versions in ``status`` (pending by default) as proposal records."""
        store = self._require_store()
        rows = await store.get_by_status(self._profile.name, status, self._config.history_limit)
        known = [row for row in rows if self._profile.get(row.name) is not None]
        active = {row.name: row for row in await store.get_all_active(self._profile.name)}
        return [self._record(row, active.get(row.name)) for row in known]

    async def version_record(self, name: str, version: int) -> dict[str, Any]:
        """One version as a proposal record: its content against the active text, with a diff."""
        self._definition(name)
        store = self._require_store()
        row = await store.get_version(self._profile.name, name, version)
        if row is None:
            raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
        return self._record(row, await store.get_active(self._profile.name, name))

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
        self,
        name: str,
        content: str,
        *,
        actor: Actor,
        expected_version: int | None = None,
        rationale: str | None = None,
    ) -> MutationResult:
        """Store a new pending version; an authorized actor approves or rejects it later."""
        return await self._write(
            name, content, actor, expected_version, activate=False, rationale=rationale
        )

    async def update(
        self, name: str, content: str, *, actor: Actor, expected_version: int | None = None
    ) -> MutationResult:
        """Store a new version and activate it at once (autonomous edit)."""
        return await self._write(name, content, actor, expected_version, activate=True)

    async def _write(
        self,
        name: str,
        content: str,
        actor: Actor,
        expected_version: int | None,
        *,
        activate: bool,
        rationale: str | None = None,
    ) -> MutationResult:
        artifact = self._definition(name)
        self._authorize(artifact, actor, "update" if activate else "propose")
        content = self._clean_content(name, content)
        async with self._require_store().transaction(self._profile.name, name) as store:
            active = await store.get_active(self._profile.name, name)
            current = active.version if active is not None else 0
            if expected_version is not None and current != expected_version:
                raise ArtifactConflictError(
                    f"Artifact '{name}' is at version {current}, not {expected_version}"
                )
            if active is not None and active.content == content:
                return self._result(active, activated=True, live=active, unchanged=True)
            row = await store.propose(
                self._profile.name,
                name,
                content,
                actor.proposed_by,
                actor_kind=actor.kind,
                rationale=(rationale or "").strip() or None,
            )
            if activate:
                row = await store.activate(self._profile.name, name, row.version)
                assert row is not None
                active = row
        if activate:
            self.invalidate()
        logger.info("Wrote artifact version", name=name, version=row.version, by=actor.kind)
        if not activate:
            await self._publish(
                {
                    "type": "artifact_proposal",
                    **self._event_ids(row),
                    "active_version": active.version if active is not None else None,
                    "rationale": row.rationale,
                    "proposed_by": {"kind": actor.kind, "label": actor.proposed_by},
                    "session_id": actor.session_id,
                }
            )
        return self._result(row, activated=activate, live=active)

    async def activate(self, name: str, version: int, *, actor: Actor) -> MutationResult:
        """Make ``version`` the active one (approval of a proposal, or rollback)."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "activate")
        async with self._require_store().transaction(self._profile.name, name) as store:
            before = await store.get_version(self._profile.name, name, version)
            row = await store.activate(
                self._profile.name, name, version, decided_by=actor.proposed_by
            )
            if before is None or row is None:
                raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
        self.invalidate()
        logger.info("Activated artifact version", name=name, version=version, by=actor.kind)
        if before.status == "pending":
            await self._publish_decision(row, actor)
        return self._result(row, activated=True, live=row)

    async def reject(
        self, name: str, version: int, *, actor: Actor, reason: str | None = None
    ) -> MutationResult:
        """Decline a pending version; the active version stays as it is."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "reject")
        async with self._require_store().transaction(self._profile.name, name) as store:
            current = await store.get_version(self._profile.name, name, version)
            if current is None:
                raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
            if current.status != "pending":
                raise ArtifactConflictError(
                    f"Version {version} of '{name}' is {current.status}, not pending"
                )
            row = await store.reject(
                self._profile.name,
                name,
                version,
                decided_by=actor.proposed_by,
                reason=(reason or "").strip() or None,
            )
            assert row is not None  # pending under the artifact's lock
            active = await store.get_active(self._profile.name, name)
        logger.info("Rejected artifact version", name=name, version=version, by=actor.kind)
        await self._publish_decision(row, actor)
        return self._result(row, activated=False, live=active)

    async def delete(self, name: str, *, actor: Actor) -> int:
        """Remove every stored version; the default text applies again."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "delete")
        async with self._require_store().transaction(self._profile.name, name) as store:
            count = await store.delete(self._profile.name, name)
        self.invalidate()
        logger.info("Deleted artifact versions", name=name, count=count, by=actor.kind)
        return count

    # --- Internals ---

    def _event_ids(self, row: ArtifactVersion) -> dict[str, Any]:
        return {
            "profile": self._profile.name,
            "subject": None,
            "name": row.name,
            "version": row.version,
        }

    async def _publish(self, event: dict[str, Any]) -> None:
        if self._owner._events is not None:
            await self._owner._events.publish(event)

    async def _publish_decision(self, row: ArtifactVersion, actor: Actor) -> None:
        await self._publish(
            {
                "type": "artifact_decision",
                **self._event_ids(row),
                "status": row.status,
                "decided_by": row.decided_by or actor.proposed_by,
                "decision_reason": row.decision_reason,
                "session_id": actor.session_id,
            }
        )

    def _record(self, row: ArtifactVersion, active: ArtifactVersion | None) -> dict[str, Any]:
        """A version as the proposal record hosts render: current and proposed, and the diff."""
        definition = self._definition(row.name)
        active_content = active.content if active is not None else definition.default
        before = (
            f"{row.name} (v{active.version})" if active is not None else f"{row.name} (default)"
        )
        diff = difflib.unified_diff(
            _lines(active_content), _lines(row.content), before, f"{row.name} (v{row.version})"
        )
        return {
            **self._event_ids(row),
            "role": definition.role,
            "status": row.status,
            "content": row.content,
            "rationale": row.rationale,
            "proposed_by": {"kind": row.actor_kind, "label": row.proposed_by},
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "active_version": active.version if active is not None else None,
            "active_content": active_content,
            "diff": "".join(diff),
            "decided_by": row.decided_by,
            "decided_at": row.decided_at.isoformat() if row.decided_at else None,
            "decision_reason": row.decision_reason,
        }

    def _require_store(self) -> ArtifactStore:
        if self._owner._store is None:
            raise ArtifactError("Artifact service not started")
        return self._owner._store

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
        if action in ("activate", "reject"):
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


def _lines(text: str) -> list[str]:
    """Lines for a diff, each ending in a newline so the last one is not glued on."""
    return (text if text.endswith("\n") else text + "\n").splitlines(keepends=True)
