"""Migration 0038 renames the steering constraints only while they keep their old names."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

MIGRATION = (
    Path(__file__).resolve().parents[4]
    / "alembic"
    / "versions"
    / "20260929_000000_steering_constraint_names.py"
)
OLD = {"pk_guidance", "ck_guidance_ck_guidance_status_valid", "fk_guidance_session_id_sessions"}
NEW = {"pk_steering", "ck_steering_ck_steering_status_valid", "fk_steering_session_id_sessions"}


class FakeConnection:
    """The steering table's constraint names; answers the existence query and applies renames."""

    def __init__(self, names: set[str]) -> None:
        self.names = set(names)
        self.renames: list[str] = []

    def execute(self, statement, params=None):
        sql = str(statement)
        if sql.startswith("ALTER TABLE"):
            old, new = sql.split('"')[1], sql.split('"')[3]
            self.names = (self.names - {old}) | {new}
            self.renames.append(sql)
            return None
        return SimpleNamespace(scalar_one=lambda: params["name"] in self.names)


@pytest.fixture
def migration():
    spec = importlib.util.spec_from_file_location("migration_0038", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(migration, monkeypatch, step: str, names: set[str]) -> FakeConnection:
    conn = FakeConnection(names)
    monkeypatch.setattr(migration, "op", SimpleNamespace(get_bind=lambda: conn))
    getattr(migration, step)()
    return conn


def test_upgrade_renames_the_guidance_names(migration, monkeypatch):
    conn = _run(migration, monkeypatch, "upgrade", OLD)
    assert conn.names == NEW
    assert len(conn.renames) == 3


def test_upgrade_leaves_model_names_alone(migration, monkeypatch):
    conn = _run(migration, monkeypatch, "upgrade", NEW)
    assert (conn.names, conn.renames) == (NEW, [])


def test_downgrade_restores_the_guidance_names(migration, monkeypatch):
    conn = _run(migration, monkeypatch, "downgrade", NEW)
    assert conn.names == OLD
