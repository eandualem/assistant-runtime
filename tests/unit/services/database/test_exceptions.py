"""Tests for database exception hierarchy."""

from lovely_assistant.base.exceptions import LovelyAssistantError
from lovely_assistant.services.database.exceptions import DatabaseConnectionError, DatabaseError


class TestDatabaseError:
    """DatabaseError base class."""

    def test_is_lovely_assistant_error(self):
        err = DatabaseError("test")
        assert isinstance(err, LovelyAssistantError)

    def test_default_category(self):
        err = DatabaseError("test")
        assert err.category == "database"

    def test_default_severity(self):
        err = DatabaseError("test")
        assert err.severity == "high"

    def test_message(self):
        err = DatabaseError("something broke")
        assert str(err) == "something broke"


class TestDatabaseConnectionError:
    """DatabaseConnectionError subclass."""

    def test_is_database_error(self):
        err = DatabaseConnectionError("connection refused")
        assert isinstance(err, DatabaseError)

    def test_default_severity(self):
        err = DatabaseConnectionError("connection refused")
        assert err.severity == "critical"

    def test_default_category(self):
        err = DatabaseConnectionError("connection refused")
        assert err.category == "database"
