"""Tests for database service factory registration."""

from unittest.mock import AsyncMock, MagicMock, patch

from assistant_runtime.services.database.factory import register_database
from assistant_runtime.services.database.interface import DatabaseService


class TestRegisterDatabase:
    """register_database() stores service on app_state and registers with lifecycle."""

    async def test_stores_service_on_app_state(self):
        app_state = MagicMock()
        lifecycle = AsyncMock()

        with patch("assistant_runtime.services.database.factory.AppSettings") as mock_settings_cls:
            mock_settings_cls.return_value = MagicMock()
            await register_database(app_state, lifecycle)

        assert hasattr(app_state, "database_service")
        assert isinstance(app_state.database_service, DatabaseService)

    async def test_registers_with_lifecycle(self):
        app_state = MagicMock()
        lifecycle = AsyncMock()

        with patch("assistant_runtime.services.database.factory.AppSettings") as mock_settings_cls:
            mock_settings_cls.return_value = MagicMock()
            await register_database(app_state, lifecycle)

        lifecycle.register.assert_awaited_once()
        call_args = lifecycle.register.call_args
        assert call_args[0][0] == "database_service"
        assert isinstance(call_args[0][1], DatabaseService)

    async def test_uses_database_config_from_settings(self):
        app_state = MagicMock()
        lifecycle = AsyncMock()

        with patch("assistant_runtime.services.database.factory.AppSettings") as mock_settings_cls:
            mock_db_config = MagicMock()
            mock_settings = MagicMock()
            mock_settings.database = mock_db_config
            mock_settings_cls.return_value = mock_settings

            await register_database(app_state, lifecycle)

            service = app_state.database_service
            assert service._config is mock_db_config
