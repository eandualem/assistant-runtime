"""Layering guard: services never import the app layer.

`base` is a leaf, `services/*` may import `base`, `config` and each other
as documented in CLAUDE.md, and only `app`, `main` and `cli` may import
`app/*`. A service that reaches into `app` would create an import cycle
through `config` (which composes every module's config) and break the
in-process CLI, so the rule is enforced here.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "assistant_runtime"
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+assistant_runtime\.app\b", re.MULTILINE)


def _offenders(package: str) -> list[str]:
    return [
        str(path.relative_to(SRC))
        for path in sorted((SRC / package).rglob("*.py"))
        if IMPORT_RE.search(path.read_text(encoding="utf-8"))
    ]


def test_services_do_not_import_app() -> None:
    assert _offenders("services") == []


def test_base_does_not_import_app_or_services() -> None:
    pattern = re.compile(
        r"^\s*(?:from|import)\s+assistant_runtime\.(?:app|services)\b", re.MULTILINE
    )
    offenders = [
        str(path.relative_to(SRC))
        for path in sorted((SRC / "base").rglob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
