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


class _PgError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.sqlstate = code


class TestFailureCause:
    def test_known_postgres_refusals_name_their_cause(self):
        from assistant_runtime.services.database.exceptions import failure_cause

        assert failure_cause(_PgError("3D000")) == "database_missing"
        assert failure_cause(_PgError("28P01")) == "auth_refused"
        assert failure_cause(_PgError("42501")) == "auth_refused"  # no CONNECT right
        assert failure_cause(ConnectionRefusedError()) == "unreachable"

    def test_the_code_is_found_on_a_cause_or_a_wrapped_original(self):
        from assistant_runtime.services.database.exceptions import failure_cause, sqlstate

        outer = RuntimeError("connect failed")
        outer.__cause__ = _PgError("3D000")
        assert failure_cause(outer) == "database_missing"
        wrapped = RuntimeError("statement failed")
        wrapped.orig = _PgError("42P04")
        assert sqlstate(wrapped) == "42P04"
