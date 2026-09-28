"""ArtifactService — registered profiles and their evolving prompt artifacts.

Public facade for the artifacts module. Implements LifecycleAware.

Every mutation goes through this class, which applies the profile's
:class:`ArtifactPolicy` from code: the artifact text itself can never
change what the assistant may do to it. With a reachable database the
versions are durable; otherwise they live in memory for the process and
every result says so (``durable=False``).

A profile view (``for_profile``) reads and writes the profile's
artifacts; a subject view (``for_subject``) also reaches the artifacts the
profile keeps per subject, for that subject.
"""

from __future__ import annotations

import difflib
import time
from collections import OrderedDict
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
    ArtifactSubjectRequiredError,
    ArtifactTooLargeError,
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
# Subject views kept per profile; the least recently used is dropped beyond this.
_SUBJECT_VIEWS = 128


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
        _as_view: bool = False,
    ) -> None:
        self._config = config
        self._profile = profile
        self._database_service = database_service
        self._events = events
        self._store: ArtifactStore | None = None
        self._cache: dict[str, str] | None = None
        self._cached_versions: dict[str, int | None] = {}
        self._cached_at = 0.0
        self._cached_generation = -1
        self._generation = 0
        """Owner only: bumped by every write, so no view serves text from before it."""
        self._started = False
        self._owner = self
        self._base = self
        self._subject = ""
        self._subject_views: OrderedDict[str, ArtifactService] = OrderedDict()
        self._views: dict[str, ArtifactService] = {profile.name: self}
        for additional in profiles:
            if additional.name in self._views:
                raise ValueError(f"Duplicate assistant profile name: {additional.name!r}")
            self._views[additional.name] = self._view(additional)
        if _as_view:
            return  # the owner checks the includes of every registered profile
        for view in self._views.values():
            for included in view._profile.include:
                if included not in self._views:
                    raise ValueError(
                        f"Profile {view._profile.name!r} includes {included!r}, "
                        "which is not registered"
                    )

    def _view(self, profile: AssistantProfile, subject: str = "") -> ArtifactService:
        view = ArtifactService(self._config, profile, self._database_service, _as_view=True)
        view._owner = self._owner
        view._views = self._owner._views
        view._subject = subject
        view._base = view if not subject else self._owner._views[profile.name]
        return view

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

    def for_subject(self, subject: str | None) -> ArtifactService:
        """This profile's view for ``subject``; no subject is the profile view itself."""
        base = self._base
        if not subject:
            return base
        view = base._subject_views.get(subject)
        if view is None:
            view = base._subject_views[subject] = self._view(base._profile, subject)
            while len(base._subject_views) > _SUBJECT_VIEWS:
                base._subject_views.popitem(last=False)  # a view holds only a text cache
        else:
            base._subject_views.move_to_end(subject)
        return view

    @property
    def subject(self) -> str | None:
        return self._subject or None

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
        self.invalidate()
        self._started = True
        logger.info(
            "Artifact service started",
            profile=self._profile.name,
            artifacts=list(self._profile.names),
            durable=self._store.durable,
        )

    async def _seed_defaults(self) -> None:
        """Store each default as a ``seed`` version where no one else has written.

        With ``DATABASE__REQUIRED`` the active text is then always a
        versioned record; the definition's default is only its seed. A
        changed default is stored and activated as a new seed only while
        every stored version is a seed, and an emptied default removes those
        seeds; any host or assistant version stops both for good.
        Subject-scoped artifacts are seeded by nobody: their subjects are not
        known in advance.
        """
        assert self._store is not None
        for view in self._views.values():
            scope = view._profile.name
            for artifact in view._profile.artifacts:
                content = (artifact.default or "").strip()
                if artifact.scope != "profile":
                    continue
                async with self._store.transaction(scope, artifact.name) as store:
                    newest = await store.get_history(scope, artifact.name, 1)
                    if newest and not await _seed_replaceable(
                        store, scope, artifact.name, newest[0], content
                    ):
                        continue
                    if not content:
                        if newest:
                            await store.delete(scope, artifact.name)
                            logger.info(
                                "Removed seeded artifact: its default is empty",
                                profile=scope,
                                name=artifact.name,
                            )
                        continue
                    row = await store.propose(scope, artifact.name, content, SEED, actor_kind=SEED)
                    await store.activate(scope, artifact.name, row.version)
                    await view._prune(store, artifact, "")
                logger.info(
                    "Seeded artifact from its default",
                    profile=scope,
                    name=artifact.name,
                    version=row.version,
                )

    async def stop(self) -> None:
        if self is not self._owner:
            await self._owner.stop()
            return
        self._store = None
        self.invalidate()
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
        """Text per declared artifact for the prompt: the active version, else the default.

        Subject-scoped artifacts take this view's subject; a profile view
        leaves them empty. Cached for ``cache_ttl_seconds``; any write
        through the service invalidates every view, so it affects the next
        prompt built.
        """
        texts, _ = await self._load_active()
        return dict(texts)

    async def prompt_inputs(
        self,
    ) -> tuple[dict[str, str], list[tuple[str, str]], dict[str, int | None]]:
        """Texts, extra fragments and versions for one prompt, from one read per profile.

        The versions are those of the texts returned, so a prompt record
        names exactly what the prompt used even while versions change.
        """
        texts, versions = await self._load_active()
        texts, versions = dict(texts), dict(versions)
        extras: list[tuple[str, str]] = []
        for name in self._profile.include:
            included = self._owner._views[name]
            included_texts, included_versions = await included._load_active()
            for artifact in included._profile.artifacts:
                if artifact.scope != "profile":
                    continue
                versions[f"{name}.{artifact.name}"] = included_versions.get(artifact.name)
                text = (included_texts.get(artifact.name) or "").strip()
                if text:
                    extras.append((f"{name}.{artifact.name}", text))
        extras.extend(await self._document_listing())
        return texts, extras, versions

    async def prompt_versions(self) -> dict[str, int | None]:
        """The version behind each artifact text of the prompt; ``None`` where the default applies.

        The profile's own artifacts by name, then the included profiles'
        as ``<profile>.<artifact>``, matching the prompt's fragments.
        """
        _, versions = await self._load_active()
        result = dict(versions)
        for name in self._profile.include:
            _, included = await self._owner._views[name]._load_active()
            for artifact in self._owner._views[name]._profile.artifacts:
                if artifact.scope == "profile":
                    result[f"{name}.{artifact.name}"] = included.get(artifact.name)
        return result

    async def _load_active(self) -> tuple[dict[str, str], dict[str, int | None]]:
        store = self._require_store()
        owner = self._owner
        now = time.monotonic()
        if (
            self._cache is not None
            and self._cached_generation == owner._generation
            and now - self._cached_at < self._config.cache_ttl_seconds
        ):
            return self._cache, self._cached_versions
        generation = owner._generation
        scope = self._profile.name
        texts = {
            a.name: a.default if a.scope == "profile" or self._subject else ""
            for a in self._profile.artifacts
        }
        versions: dict[str, int | None] = dict.fromkeys(texts)
        try:
            rows = await store.get_all_active(scope)
            if self._subject:
                rows += await store.get_all_active(scope, subject=self._subject)
            for row in rows:
                definition = self._profile.get(row.name)
                # A row counts only in its own scope: profile rows carry no subject.
                if (
                    definition is not None
                    and row.name in texts
                    and (definition.scope == "subject") == bool(row.subject)
                ):
                    texts[row.name] = row.content
                    versions[row.name] = row.version
        except Exception as e:
            # With DATABASE__REQUIRED the stored versions are the only
            # source; a prompt from defaults would silently drop them.
            if getattr(self._database_service, "required", False):
                raise ArtifactError(f"Failed to load artifacts: {e}") from e
            logger.warning("Failed to load artifacts, using defaults", error=str(e))
            return texts, dict.fromkeys(texts)
        if generation == owner._generation:
            self._cache = dict(texts)
            self._cached_versions = dict(versions)
            self._cached_generation = generation
            self._cached_at = now
        return texts, versions

    async def prompt_extras(self) -> list[tuple[str, str]]:
        """Fragments that follow the profile's own artifacts in the prompt.

        The included profiles' active profile-scoped artifacts, named
        ``<profile>.<artifact>``, then the names of the documents in the
        profile's collections (their text is read with the artifact tool).
        """
        _, extras, _ = await self.prompt_inputs()
        return extras

    async def _document_listing(self) -> list[tuple[str, str]]:
        if not self._profile.collections:
            return []
        try:
            rows = await self._require_store().get_all_active(self._profile.name)
        except Exception as e:
            # The listing is optional prompt text, like the artifacts' own fallback.
            if getattr(self._database_service, "required", False):
                raise ArtifactError(f"Failed to list documents: {e}") from e
            logger.warning("Failed to list documents, leaving them out", error=str(e))
            return []
        names = sorted(r.name for r in rows if self._profile.collection_of(r.name))
        if not names:
            return []
        return [
            (
                "documents",
                "Documents you keep (read one with manage_artifacts view): "
                + ", ".join(names)
                + ".",
            )
        ]

    def invalidate(self) -> None:
        """Forget cached texts in every view so the next prompt reads the store."""
        self._owner._generation += 1

    async def list_active(self) -> list[ArtifactVersion]:
        """Active versions of the profile's artifacts (and this subject's), in prompt order."""
        store = self._require_store()
        scope = self._profile.name
        rows = [
            row
            for row in await store.get_all_active(scope)
            if (d := self._profile.get(row.name)) is not None and d.scope == "profile"
        ]
        if self._subject:
            rows += [
                row
                for row in await store.get_all_active(scope, subject=self._subject)
                if (d := self._profile.get(row.name)) is not None and d.scope == "subject"
            ]
        return sorted(rows, key=lambda row: self._profile.sort_key(row.name))

    async def get_active(self, name: str) -> ArtifactVersion | None:
        subject = self._subject_for(self._definition(name))
        return await self._require_store().get_active(self._profile.name, name, subject=subject)

    async def history(self, name: str, limit: int | None = None) -> list[ArtifactVersion]:
        subject = self._subject_for(self._definition(name))
        return await self._require_store().get_history(
            self._profile.name, name, limit or self._config.history_limit, subject=subject
        )

    async def subjects(self) -> list[str]:
        """The subjects this profile keeps versions for."""
        return await self._require_store().get_subjects(self._profile.name)

    async def proposals(
        self, status: str = "pending", limit: int | None = None, before_id: int | None = None
    ) -> list[dict[str, Any]]:
        """The profile's versions in ``status`` (pending by default) as proposal records.

        Every subject's; newest first (by ``id``); ``limit`` bounds the page and
        ``before_id`` continues from the last ``id`` of the previous one.
        """
        store = self._require_store()
        rows = await store.get_by_status(self._profile.name, status, limit, before_id)
        records = []
        for row in rows:
            if self._profile.get(row.name) is None:
                continue
            active = await store.get_active(self._profile.name, row.name, subject=row.subject)
            records.append(self._record(row, active))
        return records

    async def version_record(self, name: str, version: int) -> dict[str, Any]:
        """One version as a proposal record: its content against the active text, with a diff."""
        subject = self._subject_for(self._definition(name))
        store = self._require_store()
        row = await store.get_version(self._profile.name, name, version, subject=subject)
        if row is None:
            raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
        return self._record(row, await store.get_active(self._profile.name, name, subject=subject))

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
        subject = self._subject_for(artifact)
        content = self._clean_content(name, content)
        if artifact.max_chars is not None and len(content) > artifact.max_chars:
            raise ArtifactTooLargeError(
                f"Artifact '{name}' holds at most {artifact.max_chars} characters and this "
                f"text has {len(content)}; condense it and write it again"
            )
        scope = self._profile.name
        async with self._require_store().transaction(scope, name, subject=subject) as store:
            active = await store.get_active(scope, name, subject=subject)
            current = active.version if active is not None else 0
            if expected_version is not None and current != expected_version:
                raise ArtifactConflictError(
                    f"Artifact '{name}' is at version {current}, not {expected_version}",
                    current_version=current,
                    current_content=active.content if active is not None else artifact.default,
                )
            if active is not None and active.content == content:
                return self._result(active, activated=True, live=active, unchanged=True)
            row = await store.propose(
                scope,
                name,
                content,
                actor.proposed_by,
                subject=subject,
                actor_kind=actor.kind,
                rationale=(rationale or "").strip() or None,
            )
            if activate:
                row = await store.activate(scope, name, row.version, subject=subject)
                assert row is not None
                active = row
                await self._prune(store, artifact, subject)
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
        subject = self._subject_for(artifact)
        scope = self._profile.name
        async with self._require_store().transaction(scope, name, subject=subject) as store:
            before = await store.get_version(scope, name, version, subject=subject)
            if before is None:
                raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
            if before.status == "rejected":
                # A decision stands; reconsidering is a new proposal with its own record.
                raise ArtifactConflictError(
                    f"Version {version} of '{name}' was rejected; propose it again to reconsider"
                )
            row = await store.activate(
                scope, name, version, subject=subject, decided_by=actor.proposed_by
            )
            assert row is not None  # the version exists under the artifact's lock
            await self._prune(store, artifact, subject)
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
        subject = self._subject_for(artifact)
        scope = self._profile.name
        async with self._require_store().transaction(scope, name, subject=subject) as store:
            current = await store.get_version(scope, name, version, subject=subject)
            if current is None:
                raise ArtifactVersionNotFoundError(f"No version {version} of artifact '{name}'")
            if current.status != "pending":
                raise ArtifactConflictError(
                    f"Version {version} of '{name}' is {current.status}, not pending"
                )
            row = await store.reject(
                scope,
                name,
                version,
                subject=subject,
                decided_by=actor.proposed_by,
                reason=(reason or "").strip() or None,
            )
            assert row is not None  # pending under the artifact's lock
            active = await store.get_active(scope, name, subject=subject)
        logger.info("Rejected artifact version", name=name, version=version, by=actor.kind)
        await self._publish_decision(row, actor)
        return self._result(row, activated=False, live=active)

    async def delete(self, name: str, *, actor: Actor) -> int:
        """Remove every stored version; the default text applies again."""
        artifact = self._definition(name)
        self._authorize(artifact, actor, "delete")
        subject = self._subject_for(artifact)
        scope = self._profile.name
        async with self._require_store().transaction(scope, name, subject=subject) as store:
            count = await store.delete(scope, name, subject=subject)
        self.invalidate()
        logger.info("Deleted artifact versions", name=name, count=count, by=actor.kind)
        return count

    # --- Internals ---

    def _subject_for(self, artifact: ArtifactDefinition) -> str:
        """The stored subject of ``artifact`` in this view: empty unless it is kept per subject."""
        if artifact.scope != "subject":
            return ""
        if not self._subject:
            raise ArtifactSubjectRequiredError(
                f"Artifact '{artifact.name}' is kept per subject; name the subject"
            )
        return self._subject

    async def _prune(
        self, store: ArtifactStore, artifact: ArtifactDefinition, subject: str
    ) -> None:
        if artifact.keep_versions is not None:
            await store.prune_superseded(
                self._profile.name, artifact.name, artifact.keep_versions, subject=subject
            )

    def _event_ids(self, row: ArtifactVersion) -> dict[str, Any]:
        return {
            "profile": self._profile.name,
            "subject": row.subject or None,
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
            "id": row.id,
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


async def _seed_replaceable(
    store: Any, scope: str, name: str, newest: ArtifactVersion, content: str
) -> bool:
    """Whether a changed default may replace the stored seed.

    Only while every stored version is a seed and the newest is active, so
    no one's edit or rollback is ever overwritten. Retention never deletes
    the newest version, so after a host or assistant write the newest is
    theirs for good.
    """
    if newest.actor_kind != SEED or not newest.is_active or newest.content == content:
        return False
    history = await store.get_history(scope, name, newest.version)
    return all(v.actor_kind == SEED for v in history)
