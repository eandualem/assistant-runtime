"""Tests for SQLAlchemy declarative base and naming conventions."""

from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase

from assistant_runtime.services.database.base import Base, convention


class TestBase:
    """DeclarativeBase configuration."""

    def test_is_declarative_base(self):
        assert issubclass(Base, DeclarativeBase)

    def test_has_metadata(self):
        assert isinstance(Base.metadata, MetaData)


class TestNamingConventions:
    """Naming convention completeness."""

    def test_convention_has_index(self):
        assert "ix" in convention

    def test_convention_has_unique(self):
        assert "uq" in convention

    def test_convention_has_check(self):
        assert "ck" in convention

    def test_convention_has_foreign_key(self):
        assert "fk" in convention

    def test_convention_has_primary_key(self):
        assert "pk" in convention

    def test_metadata_uses_convention(self):
        nc = Base.metadata.naming_convention
        for key in ("ix", "uq", "ck", "fk", "pk"):
            assert key in nc
