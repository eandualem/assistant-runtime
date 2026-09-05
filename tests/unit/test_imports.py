"""Layering guard: services never import the app layer.

`base` and `artifacts` are leaves, `services/*` may import `base`, `config`,
`artifacts` and each other as documented in AGENTS.md, and only the top
layer (`app`, `main`, `cli`, and `config`, which composes every module's
config model) may import `app/*`. A service that reaches into `app` would create
an import cycle through `config` (which composes every module's config) and
break the in-process CLI, so the rule is enforced here.

Imports are read with `ast` and relative imports are resolved, so
``from ..app import x``, ``from assistant_runtime import app`` and
``import assistant_runtime.app.routes`` are all caught.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

PACKAGE = "assistant_runtime"
SRC = Path(__file__).resolve().parents[2] / "src" / PACKAGE


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imported_modules(path: Path) -> set[str]:
    """Every module an import statement in ``path`` can resolve to, absolute."""
    module = _module_name(path)
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            found.add(base)
            # ``from assistant_runtime import app`` names a submodule, not an attribute
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _offenders(subpackage: str, forbidden: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for path in sorted((SRC / subpackage).rglob("*.py")):
        hits = sorted(
            imported
            for imported in _imported_modules(path)
            if any(imported == f or imported.startswith(f + ".") for f in forbidden)
        )
        if hits:
            result.append(f"{path.relative_to(SRC)}: {', '.join(hits)}")
    return result


# Modules allowed to import assistant_runtime.app: the app package itself, the
# entry points, and config (it composes the module config models).
APP_IMPORTERS = ("app", "main", "cli", "config")


def _all_offenders(forbidden: tuple[str, ...], allowed_top_level: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        top = path.relative_to(SRC).parts[0].removesuffix(".py")
        if top in allowed_top_level:
            continue
        hits = sorted(
            imported
            for imported in _imported_modules(path)
            if any(imported == f or imported.startswith(f + ".") for f in forbidden)
        )
        if hits:
            result.append(f"{path.relative_to(SRC)}: {', '.join(hits)}")
    return result


def test_only_the_top_layer_imports_app() -> None:
    assert _all_offenders((f"{PACKAGE}.app",), APP_IMPORTERS) == []


def test_services_do_not_import_app() -> None:
    assert _offenders("services", (f"{PACKAGE}.app",)) == []


def test_base_does_not_import_app_or_services() -> None:
    assert _offenders("base", (f"{PACKAGE}.app", f"{PACKAGE}.services")) == []


@pytest.mark.parametrize("leaf", ["artifacts.py", "model_catalog.py"])
def test_leaf_modules_import_nothing_from_the_package(leaf: str) -> None:
    project_imports = {m for m in _imported_modules(SRC / leaf) if m.startswith(PACKAGE)}
    assert project_imports == set()


class TestResolver:
    """The resolver itself, on synthetic files, so a regression cannot hide."""

    @pytest.fixture
    def check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "src" / PACKAGE)

        def _check(rel: str, source: str) -> set[str]:
            file = tmp_path / "src" / PACKAGE / rel
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(source)
            return _imported_modules(file)

        return _check

    def test_relative_import_resolves(self, check) -> None:
        found = check("services/tools/_x.py", "from ...app.assistant import y\n")
        assert f"{PACKAGE}.app.assistant" in found

    def test_package_root_import_resolves(self, check) -> None:
        found = check("services/tools/_x.py", f"from {PACKAGE} import app\n")
        assert f"{PACKAGE}.app" in found

    def test_plain_import_resolves(self, check) -> None:
        found = check("services/tools/_x.py", f"import {PACKAGE}.app.routes\n")
        assert f"{PACKAGE}.app.routes" in found

    def test_init_relative_import_resolves(self, check) -> None:
        found = check("services/tools/__init__.py", "from ..llm import z\n")
        assert f"{PACKAGE}.services.llm" in found
