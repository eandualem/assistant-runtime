"""Tests for database exception hierarchy."""

from assistant_runtime.base.exceptions import AssistantRuntimeError
from assistant_runtime.services.database.exceptions import DatabaseError


class TestDatabaseError:
    """DatabaseError base class."""

    def test_is_assistant_runtime_error(self):
        err = DatabaseError("test")
        assert isinstance(err, AssistantRuntimeError)

    def test_default_category(self):
        err = DatabaseError("test")
        assert err.category == "database"

    def test_default_severity(self):
        err = DatabaseError("test")
        assert err.severity == "high"

    def test_message(self):
        err = DatabaseError("something broke")
        assert str(err) == "something broke"
