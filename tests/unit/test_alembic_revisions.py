from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSIONS_DIR = ROOT / "alembic" / "versions"


def _read_revision_metadata(path: Path) -> tuple[str, str | tuple[str, ...] | None]:
    module = ast.parse(path.read_text(), filename=str(path))
    revision: str | None = None
    down_revision: str | tuple[str, ...] | None = None

    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "revision" in targets:
                revision = ast.literal_eval(node.value)
            if "down_revision" in targets:
                down_revision = ast.literal_eval(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "revision":
                revision = ast.literal_eval(node.value)
            if node.target.id == "down_revision":
                down_revision = ast.literal_eval(node.value)

    if revision is None:
        raise AssertionError(f"Migration {path.name} is missing a revision id")

    return revision, down_revision


def test_alembic_revisions_are_unique_and_linear() -> None:
    revisions: dict[str, Path] = {}
    referenced: set[str] = set()

    for path in sorted(VERSIONS_DIR.glob("*.py")):
        revision, down_revision = _read_revision_metadata(path)
        assert revision not in revisions, (
            f"Duplicate Alembic revision {revision} in {revisions[revision].name} and {path.name}"
        )
        revisions[revision] = path

        if isinstance(down_revision, tuple):
            referenced.update(down_revision)
        elif isinstance(down_revision, str):
            referenced.add(down_revision)

    heads = sorted(set(revisions) - referenced)
    assert len(heads) == 1, f"Expected one Alembic head, found {heads}"
