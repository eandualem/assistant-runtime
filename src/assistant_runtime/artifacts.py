"""Assistant profiles: the prompt artifacts an assistant is made of.

A leaf module (no project imports) so the app layer (prompt builder,
definitions), the artifact service and the tools layer share one schema.

An :class:`AssistantProfile` names its artifacts, their prompt order, their
roles, the text used until a version is activated, and what the assistant
and the host may do to each of them. Profiles come from Python
(``AssistantDefinition.profile``), from a TOML file (``ASSISTANT__PROFILE``
pointing at it), or from the built-in ones (``neutral`` and the
``technical_operator`` example).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EditPolicy = Literal["none", "propose", "autonomous"]
_EDIT_POLICIES: tuple[str, ...] = ("none", "propose", "autonomous")
ArtifactScope = Literal["profile", "subject"]
_SCOPES: tuple[str, ...] = ("profile", "subject")
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,30}_$")
PROFILES_DIR = Path(__file__).with_name("profiles")
BUILTIN_PROFILE_NAMES: tuple[str, ...] = ("neutral", "technical_operator")


@dataclass(frozen=True)
class ArtifactPolicy:
    """What each actor may do to an artifact. Enforced in code, never by prompt text."""

    assistant_edit: EditPolicy = "propose"
    """``none``: read only; ``propose``: new versions stay inactive until an
    authorized actor activates them; ``autonomous``: the assistant's writes
    activate immediately."""
    assistant_activate: bool = False
    """Whether the assistant may activate or roll back versions itself."""
    host_edit: bool = True
    """Whether the host application (HTTP routes) may write, activate, roll back and delete."""

    def __post_init__(self) -> None:
        if self.assistant_edit not in _EDIT_POLICIES:
            raise ValueError(
                f"assistant_edit must be one of {', '.join(_EDIT_POLICIES)}: {self.assistant_edit!r}"
            )
        # Policies decide authorization; a quoted "false" must not read as true.
        for name in ("assistant_activate", "host_edit"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean: {getattr(self, name)!r}")


@dataclass(frozen=True)
class ArtifactDefinition:
    """One named prompt fragment: its role, default text and mutation policy."""

    name: str
    role: str = ""
    required: bool = False
    """The prompt cannot be built while the artifact has no text."""
    default: str = ""
    """Text used until a version is activated; may be empty for optional artifacts."""
    policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    scope: ArtifactScope = "profile"
    """``profile``: one text for every turn of the profile. ``subject``: one text
    per subject (for example, per agent an assistant keeps track of), chosen
    by the request's ``subject``; without a subject it is left out."""
    keep_versions: int | None = None
    """Superseded versions kept per artifact and subject; ``None`` keeps all."""
    max_chars: int | None = None
    """Longest text a write may store; a longer one is refused. ``None`` sets no bound."""

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Artifact name must be lowercase letters, digits and underscores: {self.name!r}"
            )
        if self.required and not self.default.strip():
            raise ValueError(f"Required artifact {self.name!r} needs a non-empty default")
        if self.scope not in _SCOPES:
            raise ValueError(f"Artifact {self.name!r} scope must be profile or subject")
        if self.required and self.scope == "subject":
            raise ValueError(f"Subject-scoped artifact {self.name!r} cannot be required")
        _check_bound(self.name, "keep_versions", self.keep_versions)
        _check_bound(self.name, "max_chars", self.max_chars)
        if self.max_chars is not None and len(self.default.strip()) > self.max_chars:
            raise ValueError(f"Artifact {self.name!r} default is longer than max_chars")


@dataclass(frozen=True)
class ArtifactCollection:
    """Documents the assistant may create itself, named ``<prefix><anything>``.

    A collection's documents are versioned artifacts with the collection's
    role and policy; the prompt lists their names instead of including
    their text, and the assistant reads one with the artifact tool.
    """

    prefix: str
    role: str = ""
    policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    keep_versions: int | None = None
    max_chars: int | None = None

    def __post_init__(self) -> None:
        if not _PREFIX_PATTERN.match(self.prefix):
            raise ValueError(
                "Collection prefix must be lowercase letters, digits and underscores, "
                f"ending in '_': {self.prefix!r}"
            )
        _check_bound(self.prefix, "keep_versions", self.keep_versions)
        _check_bound(self.prefix, "max_chars", self.max_chars)

    def matches(self, name: str) -> bool:
        return (
            name.startswith(self.prefix)
            and len(name) > len(self.prefix)
            and bool(_NAME_PATTERN.match(name))
        )

    def definition(self, name: str) -> ArtifactDefinition:
        return ArtifactDefinition(
            name=name,
            role=self.role,
            policy=self.policy,
            keep_versions=self.keep_versions,
            max_chars=self.max_chars,
        )


def _check_bound(owner: str, field_name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError(f"{owner!r} {field_name} must be a positive integer or None")


@dataclass(frozen=True)
class AssistantProfile:
    """The ordered artifacts of one assistant, plus the store scope they live in.

    ``name`` scopes stored versions: two profiles with different names never
    see each other's artifacts, even when artifact names coincide. The
    built-ins use ``neutral`` and ``technical_operator``; a file profile
    without a name is ``default``.
    """

    name: str = "default"
    artifacts: tuple[ArtifactDefinition, ...] = ()
    include: tuple[str, ...] = ()
    """Other registered profiles whose active artifacts follow this profile's
    own in the prompt (their profile-scoped artifacts only), for text several
    profiles share."""
    collections: tuple[ArtifactCollection, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Profile name must be lowercase letters, digits and underscores: {self.name!r}"
            )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "include", tuple(self.include))
        object.__setattr__(self, "collections", tuple(self.collections))
        seen: set[str] = set()
        for artifact in self.artifacts:
            if artifact.name in seen:
                raise ValueError(f"Duplicate artifact name in profile: {artifact.name!r}")
            seen.add(artifact.name)
        for included in self.include:
            if not _NAME_PATTERN.match(included) or included == self.name:
                raise ValueError(f"Profile {self.name!r} cannot include {included!r}")
        prefixes = [collection.prefix for collection in self.collections]
        for prefix in prefixes:
            if any(other != prefix and other.startswith(prefix) for other in prefixes):
                raise ValueError(f"Collection prefix {prefix!r} overlaps another")
            if any(name.startswith(prefix) for name in seen):
                raise ValueError(f"Collection prefix {prefix!r} matches a declared artifact")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("Duplicate collection prefix in profile")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.artifacts)

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(artifact.name for artifact in self.artifacts if artifact.required)

    @property
    def defaults(self) -> dict[str, str]:
        """Default text per artifact, the prompt input when nothing is activated."""
        return {artifact.name: artifact.default for artifact in self.artifacts}

    def get(self, name: str) -> ArtifactDefinition | None:
        """The declared artifact, else a collection document of that name."""
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        for collection in self.collections:
            if collection.matches(name):
                return collection.definition(name)
        return None

    def collection_of(self, name: str) -> ArtifactCollection | None:
        return next((c for c in self.collections if c.matches(name)), None)

    def sort_key(self, name: str) -> tuple[int, str]:
        """Prompt order first, then unknown names alphabetically."""
        for index, artifact in enumerate(self.artifacts):
            if artifact.name == name:
                return (index, name)
        return (len(self.artifacts), name)

    def names_text(self) -> str:
        return ", ".join(self.names)


# --- Built-in profiles ---


def neutral_profile() -> AssistantProfile:
    """The default: one required instructions artifact and an autonomous scratchpad."""
    return AssistantProfile(
        name="neutral",
        artifacts=(
            ArtifactDefinition(
                name="instructions",
                role="what the assistant is for and how it should behave",
                required=True,
                default=(
                    "You are the assistant of this application. Answer clearly and "
                    "concisely, use the available tools when they help, and say so "
                    "when you are not sure."
                ),
            ),
            ArtifactDefinition(
                name="scratchpad",
                role="short-lived operational memory the assistant keeps for itself",
                policy=ArtifactPolicy(assistant_edit="autonomous"),
            ),
        ),
    )


def technical_operator_profile() -> AssistantProfile:
    """The original technical-operator assistant, kept as an example profile."""
    directory = PROFILES_DIR / "technical_operator"
    roles = {
        "soul": "enduring purpose, values, non-negotiables, deepest identity guidance",
        "persona": "style, stance, behavioral voice",
        "communication_protocol": "interaction and routing rules",
        "ecosystem": "world model, roles, org structure, system topology",
    }
    return AssistantProfile(
        # Rows stored before profiles existed belong to this assistant (migration 0020).
        name="technical_operator",
        artifacts=(
            *(
                ArtifactDefinition(
                    name=name,
                    role=role,
                    required=True,
                    default=(directory / f"{name}.md").read_text(encoding="utf-8").strip(),
                )
                for name, role in roles.items()
            ),
            ArtifactDefinition(
                name="scratchpad",
                role="short-lived operational memory",
                policy=ArtifactPolicy(assistant_edit="autonomous"),
            ),
        ),
    )


_BUILTIN_PROFILES = {
    "neutral": neutral_profile,
    "technical_operator": technical_operator_profile,
}


# --- File-based profiles ---


def load_profile_file(path: str | Path) -> AssistantProfile:
    """Read a profile from a TOML file.

    ``name`` scopes the stored artifact versions and must match
    ``[a-z][a-z0-9_]{0,63}`` (lowercase, digits, underscores; no hyphens).

    ::

        name = "support"

        [[artifacts]]
        name = "instructions"
        role = "what the assistant is for"
        required = true
        default_file = "instructions.md"   # relative to this file, or `default = "..."`

        [artifacts.policy]
        assistant_edit = "propose"         # none | propose | autonomous
        assistant_activate = false
        host_edit = true

    An artifact may also set ``scope = "subject"`` (one text per request
    ``subject``), ``keep_versions`` and ``max_chars``. A profile may
    ``include = ["owner"]`` (other registered profiles' artifacts follow its
    own in the prompt) and declare ``[[collections]]`` with a ``prefix`` (for
    example ``"doc_"``), ``role``, ``policy``, ``keep_versions`` and
    ``max_chars``: documents the assistant creates itself.
    """
    path = Path(path)
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"Profile file not found: {path}") from None
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Profile file {path} is not valid TOML: {e}") from e
    return profile_from_mapping(data, base_dir=path.parent)


def profile_from_mapping(data: dict[str, Any], *, base_dir: Path | None = None) -> AssistantProfile:
    """Build a profile from the mapping shape used by the TOML file."""
    if not isinstance(data, dict):
        raise ValueError("Profile must be a table")
    artifacts_data = data.get("artifacts", [])
    if not isinstance(artifacts_data, list):
        raise ValueError("Profile 'artifacts' must be an array of tables")
    artifacts: list[ArtifactDefinition] = []
    for item in artifacts_data:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("Each artifact needs at least a 'name'")
        default = item.get("default", "")
        default_file = item.get("default_file")
        if default_file:
            file_path = Path(default_file)
            if not file_path.is_absolute():
                file_path = (base_dir or Path.cwd()) / file_path
            try:
                default = file_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                raise ValueError(
                    f"Artifact {item['name']!r} default_file not found: {file_path}"
                ) from None
        policy_data = item.get("policy", {})
        if not isinstance(policy_data, dict):
            raise ValueError(f"Artifact {item['name']!r} policy must be a table")
        artifacts.append(
            ArtifactDefinition(
                name=str(item["name"]),
                role=str(item.get("role", "")),
                required=bool(item.get("required", False)),
                default=str(default).strip(),
                policy=ArtifactPolicy(**policy_data),
                scope=item.get("scope", "profile"),
                keep_versions=item.get("keep_versions"),
                max_chars=item.get("max_chars"),
            )
        )
    collections_data = data.get("collections", [])
    if not isinstance(collections_data, list):
        raise ValueError("Profile 'collections' must be an array of tables")
    collections: list[ArtifactCollection] = []
    for item in collections_data:
        if not isinstance(item, dict) or "prefix" not in item:
            raise ValueError("Each collection needs a 'prefix'")
        policy_data = item.get("policy", {})
        if not isinstance(policy_data, dict):
            raise ValueError(f"Collection {item['prefix']!r} policy must be a table")
        collections.append(
            ArtifactCollection(
                prefix=str(item["prefix"]),
                role=str(item.get("role", "")),
                policy=ArtifactPolicy(**policy_data),
                keep_versions=item.get("keep_versions"),
                max_chars=item.get("max_chars"),
            )
        )
    include = data.get("include", [])
    if not isinstance(include, list) or not all(isinstance(n, str) for n in include):
        raise ValueError("Profile 'include' must be an array of profile names")
    return AssistantProfile(
        name=str(data.get("name", "default")),
        artifacts=tuple(artifacts),
        include=tuple(include),
        collections=tuple(collections),
    )


def resolve_profile(
    setting: str | None = None, definition_profile: AssistantProfile | None = None
) -> AssistantProfile:
    """Pick the profile: the definition's, else ``ASSISTANT__PROFILE``, else ``neutral``.

    ``setting`` is a built-in profile name or the path of a TOML profile file.
    """
    if definition_profile is not None:
        return definition_profile
    if not setting:
        return neutral_profile()
    builder = _BUILTIN_PROFILES.get(setting)
    if builder is not None:
        return builder()
    return load_profile_file(setting)


__all__ = [
    "BUILTIN_PROFILE_NAMES",
    "PROFILES_DIR",
    "ArtifactDefinition",
    "ArtifactPolicy",
    "AssistantProfile",
    "EditPolicy",
    "load_profile_file",
    "neutral_profile",
    "profile_from_mapping",
    "resolve_profile",
    "technical_operator_profile",
]
