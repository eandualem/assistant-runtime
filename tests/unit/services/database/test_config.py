"""Tests for database service configuration."""

import pytest
from pydantic import ValidationError

from lovely_assistant.services.database.config import DatabaseConfig


class TestDatabaseConfigDefaults:
    """Default values and basic construction."""

    def test_defaults(self):
        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5434
        assert config.user == "lovely_assistant"
        assert config.password == "lovely_assistant"
        assert config.name == "lovely_assistant"
        assert config.pool_size == 5
        assert config.pool_overflow == 10
        assert config.echo is False

    def test_custom_values(self):
        config = DatabaseConfig(
            host="db.example.com",
            port=5432,
            user="custom_user",
            password="secret123",
            name="custom_db",
            pool_size=10,
            pool_overflow=20,
            echo=True,
        )
        assert config.host == "db.example.com"
        assert config.port == 5432
        assert config.user == "custom_user"
        assert config.password == "secret123"
        assert config.name == "custom_db"
        assert config.pool_size == 10
        assert config.pool_overflow == 20
        assert config.echo is True

    def test_frozen(self):
        config = DatabaseConfig()
        with pytest.raises(ValidationError):
            config.host = "other"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(unknown="value")


class TestDatabaseConfigComputedUrls:
    """Computed URL properties."""

    def test_async_url_default(self):
        config = DatabaseConfig()
        assert config.async_url == (
            "postgresql+asyncpg://lovely_assistant:lovely_assistant@localhost:5434/lovely_assistant"
        )

    def test_sync_url_default(self):
        config = DatabaseConfig()
        assert config.sync_url == (
            "postgresql://lovely_assistant:lovely_assistant@localhost:5434/lovely_assistant"
        )

    def test_async_url_custom(self):
        config = DatabaseConfig(
            host="db.prod",
            port=5432,
            user="admin",
            password="s3cret",
            name="production",
        )
        assert config.async_url == "postgresql+asyncpg://admin:s3cret@db.prod:5432/production"

    def test_sync_url_custom(self):
        config = DatabaseConfig(
            host="db.prod",
            port=5432,
            user="admin",
            password="s3cret",
            name="production",
        )
        assert config.sync_url == "postgresql://admin:s3cret@db.prod:5432/production"

    def test_async_url_uses_asyncpg_driver(self):
        config = DatabaseConfig()
        assert config.async_url.startswith("postgresql+asyncpg://")

    def test_sync_url_uses_plain_postgresql(self):
        config = DatabaseConfig()
        assert config.sync_url.startswith("postgresql://")
        assert "+asyncpg" not in config.sync_url


class TestDatabaseConfigPoolValidation:
    """Pool configuration boundary validation."""

    def test_pool_size_minimum(self):
        config = DatabaseConfig(pool_size=1)
        assert config.pool_size == 1

    def test_pool_size_maximum(self):
        config = DatabaseConfig(pool_size=20)
        assert config.pool_size == 20

    def test_pool_size_below_minimum(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=0)

    def test_pool_size_above_maximum(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_size=21)

    def test_pool_overflow_minimum(self):
        config = DatabaseConfig(pool_overflow=0)
        assert config.pool_overflow == 0

    def test_pool_overflow_maximum(self):
        config = DatabaseConfig(pool_overflow=40)
        assert config.pool_overflow == 40

    def test_pool_overflow_below_minimum(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_overflow=-1)

    def test_pool_overflow_above_maximum(self):
        with pytest.raises(ValidationError):
            DatabaseConfig(pool_overflow=41)

    def test_port_as_int(self):
        config = DatabaseConfig(port=5432)
        assert isinstance(config.port, int)
