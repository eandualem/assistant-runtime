"""Which providers are configured, read from conventional environment variables.

A provider is an integration that serves one or more capabilities (a notes
folder, an issue tracker, a messaging channel, an agent orchestrator). Each
one is enabled by its own variables; nothing is assumed. The capabilities a
configured provider serves are registered as tools, the others are not
offered to the model at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


def _parse_named_paths(value: str) -> dict[str, Path]:
    """``name=path,name=path`` (or a bare path, named ``default``) into a mapping."""
    roots: dict[str, Path] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, path = item.partition("=")
        if not sep:
            name, path = "default", name
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise ValueError(f"LIBRARY_PATHS entry '{item}' needs both a name and a path")
        roots[name] = Path(path).expanduser()
    return roots


class ProvidersConfig(BaseModel):
    """The configured providers. Built by ``from_env``; nested in ``AppSettings`` as ``providers``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    notes_path: Path | None = Field(
        default=None,
        description="Folder of markdown notes the assistant manages (NOTES_PATH).",
    )
    library_paths: dict[str, Path] = Field(
        default_factory=dict,
        description=(
            "Named directories of reference documents the assistant can read "
            "(LIBRARY_PATHS as name=path,name=path)."
        ),
    )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProvidersConfig:
        env = os.environ if environ is None else environ
        notes = env.get("NOTES_PATH", "").strip()
        return cls(
            notes_path=Path(notes).expanduser() if notes else None,
            library_paths=_parse_named_paths(env.get("LIBRARY_PATHS", "")),
        )

    def configured(self) -> list[str]:
        """Names of the providers that are configured, for the doctor and logs."""
        names = []
        if self.notes_path is not None:
            names.append("notes")
        if self.library_paths:
            names.append("library")
        return names
