"""The documentation, readable from the installed package.

``docs/*.md`` in a source checkout is force-included into the wheel as
``assistant_runtime/help/docs`` so ``assistant-runtime docs <page>`` works
without a checkout of the repository. Whichever location exists is used.
"""

from __future__ import annotations

import re
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,40}$")
_DOCS_DIRS = (
    Path(__file__).with_name("docs"),
    Path(__file__).resolve().parents[3] / "docs",
)
_INDEX_PAGE = "README"


def docs_dir() -> Path | None:
    """Where the pages are, or None when nothing is shipped or checked out."""
    return next((d for d in _DOCS_DIRS if d.is_dir()), None)


def _summary(path: Path) -> str:
    """The first non-empty line without its heading marker."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return line.lstrip("# ").strip()
    return ""


def list_docs() -> list[dict[str, str]]:
    """The pages as ``{name, summary}``, sorted by name; the index page is excluded."""
    directory = docs_dir()
    if directory is None:
        return []
    return [
        {"name": path.stem, "summary": _summary(path)}
        for path in sorted(directory.glob("*.md"))
        if _NAME_RE.match(path.stem) and path.stem != _INDEX_PAGE
    ]


def get_doc(name: str) -> str | None:
    """One page's markdown, or None when the name is unknown."""
    directory = docs_dir()
    if directory is None or not _NAME_RE.match(name):
        return None
    path = directory / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.is_file() else None
