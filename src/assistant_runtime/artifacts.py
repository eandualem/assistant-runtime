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
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EditPolicy = Literal["none", "propose", "autonomous"]
_EDIT_POLICIES: tuple[str, ...] = ("none", "propose", "autonomous")
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
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

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Artifact name must be lowercase letters, digits and underscores: {self.name!r}"
            )
        if self.required and not self.default.strip():
            raise ValueError(f"Required artifact {self.name!r} needs a non-empty default")


@dataclass(frozen=True)
class AssistantProfile:
    """The ordered artifacts of one assistant, plus the store scope they live in.

    ``name`` scopes stored versions: two profiles with different names never
    see each other's artifacts, even when artifact names coincide.
    """

    name: str = "default"
    artifacts: tuple[ArtifactDefinition, ...] = ()

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"Profile name must be lowercase letters, digits and underscores: {self.name!r}"
            )
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        seen: set[str] = set()
        for artifact in self.artifacts:
            if artifact.name in seen:
                raise ValueError(f"Duplicate artifact name in profile: {artifact.name!r}")
            seen.add(artifact.name)

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
        for artifact in self.artifacts:
            if artifact.name == name:
                return artifact
        return None

    def sort_key(self, name: str) -> tuple[int, str]:
        """Prompt order first, then unknown names alphabetically."""
        for index, artifact in enumerate(self.artifacts):
            if artifact.name == name:
                return (index, name)
        return (len(self.artifacts), name)

    def names_text(self) -> str:
        return ", ".join(self.names)

    def roles_text(self) -> str:
        return "; ".join(f"{a.name} = {a.role}" for a in self.artifacts if a.role)


# --- Built-in profiles ---


def neutral_profile() -> AssistantProfile:
    """The default: one required instructions artifact and an autonomous scratchpad."""
    return AssistantProfile(
        name="default",
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
        name="default",
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
            )
        )
    return AssistantProfile(name=str(data.get("name", "default")), artifacts=tuple(artifacts))


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


def profile_names(profiles: Iterable[str] = BUILTIN_PROFILE_NAMES) -> str:
    return ", ".join(profiles)


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
