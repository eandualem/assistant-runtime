"""Bundled default prompt artifacts.

The system prompt is assembled from four artifacts (soul, persona,
communication_protocol, ecosystem). Installations with a database keep
versioned copies in the ``artifacts`` table; without one, or when the table
has no active row for a name, the files in ``defaults/`` are used so the
runtime can answer with nothing configured beyond a provider key.
"""

from __future__ import annotations

from pathlib import Path

from assistant_runtime.app.assistant._prompt_builder import REQUIRED_ARTIFACT_NAMES

DEFAULTS_DIR = Path(__file__).with_name("defaults")


def load_default_artifacts() -> dict[str, str]:
    """Return the bundled artifact texts keyed by artifact name."""
    return {
        name: (DEFAULTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
        for name in REQUIRED_ARTIFACT_NAMES
    }
