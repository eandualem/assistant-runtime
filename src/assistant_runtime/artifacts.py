"""The prompt artifact catalog: names, roles and ordering.

A leaf module (no project imports) so both the app layer (prompt builder,
defaults) and the tool layer (the manage_artifacts tool) can share it
without the tools package importing `app`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactDefinition:
    """Definition of a first-class prompt artifact."""

    name: str
    role_boundary: str
    required: bool
    scratchpad_special_case: bool = False


ARTIFACT_CATALOG: tuple[ArtifactDefinition, ...] = (
    ArtifactDefinition(
        name="soul",
        role_boundary="enduring purpose, values, non-negotiables, deepest identity guidance",
        required=True,
    ),
    ArtifactDefinition(
        name="persona",
        role_boundary="style, stance, behavioral voice",
        required=True,
    ),
    ArtifactDefinition(
        name="communication_protocol",
        role_boundary="interaction and routing rules",
        required=True,
    ),
    ArtifactDefinition(
        name="ecosystem",
        role_boundary="world model, roles, org structure, system topology",
        required=True,
    ),
    ArtifactDefinition(
        name="scratchpad",
        role_boundary="short-lived operational memory",
        required=False,
        scratchpad_special_case=True,
    ),
)
_ARTIFACTS_BY_NAME: dict[str, ArtifactDefinition] = {item.name: item for item in ARTIFACT_CATALOG}
_REQUIRED_ARTIFACTS = tuple(item.name for item in ARTIFACT_CATALOG if item.required)
REQUIRED_ARTIFACT_NAMES = _REQUIRED_ARTIFACTS
KNOWN_ARTIFACT_NAMES = tuple(item.name for item in ARTIFACT_CATALOG)
SCRATCHPAD_ARTIFACT_NAME = next(
    item.name for item in ARTIFACT_CATALOG if item.scratchpad_special_case
)
_ARTIFACT_ORDER = {item.name: idx for idx, item in enumerate(ARTIFACT_CATALOG)}


def artifact_sort_key(name: str) -> tuple[int, str]:
    """Sort first-class artifacts in canonical prompt/dashboard order."""
    return (_ARTIFACT_ORDER.get(name, len(ARTIFACT_CATALOG)), name)


def is_known_artifact_name(name: str) -> bool:
    """Return whether the name is a first-class artifact."""
    return name in _ARTIFACTS_BY_NAME


def known_artifact_names_text() -> str:
    """Human-readable list of known first-class artifact names."""
    return ", ".join(KNOWN_ARTIFACT_NAMES)


def artifact_role_boundaries_text() -> str:
    """Human-readable summary of the artifact role split."""
    return "; ".join(f"{item.name} = {item.role_boundary}" for item in ARTIFACT_CATALOG)
