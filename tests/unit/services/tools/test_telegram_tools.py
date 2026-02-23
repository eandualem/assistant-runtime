"""Tests for Telegram messaging tools."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._telegram_tools import (
    register_telegram_tools,
    respond_telegram,
)
from lovely_assistant.services.tools.config import ToolConfig


class TestRespondTelegram:
    """Tests for the respond_telegram tool handler."""

    async def test_empty_message_returns_error(self):
        result = await respond_telegram("")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_whitespace_only_message_returns_error(self):
        result = await respond_telegram("   ")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    async def test_missing_token_returns_auth_error(self):
        with patch.dict("os.environ", {}, clear=True):
            result = await respond_telegram("Hello Elias")
        assert result["success"] is False
        assert result["error_code"] == "TELEGRAM_AUTH_MISSING"
        assert "TELEGRAM_TOKEN" in result["error"]

    async def test_success_returns_message_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "result": {"message_id": 42},
        }

        with (
            patch.dict("os.environ", {"TELEGRAM_TOKEN": "test-token"}, clear=True),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await respond_telegram("Hello Elias")

        assert result["success"] is True
        assert result["message_id"] == 42

    async def test_success_uses_default_chat_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 1}}

        with (
            patch.dict("os.environ", {"TELEGRAM_TOKEN": "test-token"}, clear=True),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await respond_telegram("Hello")

            # Verify the call used the default chat ID
            call_kwargs = mock_client.post.call_args
            assert call_kwargs[1]["json"]["chat_id"] == "897573812"

    async def test_success_uses_custom_chat_id(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 1}}

        with (
            patch.dict(
                "os.environ",
                {"TELEGRAM_TOKEN": "test-token", "TELEGRAM_CHAT_ID": "999"},
                clear=True,
            ),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await respond_telegram("Hello")

            call_kwargs = mock_client.post.call_args
            assert call_kwargs[1]["json"]["chat_id"] == "999"

    async def test_api_error_returns_failure(self):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "ok": False,
            "description": "Forbidden: bot was blocked by the user",
        }

        with (
            patch.dict("os.environ", {"TELEGRAM_TOKEN": "test-token"}, clear=True),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await respond_telegram("Hello")

        assert result["success"] is False
        assert result["error_code"] == "TELEGRAM_API_ERROR"
        assert "403" in result["error"]
        assert "Forbidden" in result["error"]

    async def test_timeout_returns_timeout_error(self):
        with (
            patch.dict("os.environ", {"TELEGRAM_TOKEN": "test-token"}, clear=True),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value = mock_client

            result = await respond_telegram("Hello")

        assert result["success"] is False
        assert result["error_code"] == "TELEGRAM_TIMEOUT"

    async def test_strips_message_whitespace(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "result": {"message_id": 1}}

        with (
            patch.dict("os.environ", {"TELEGRAM_TOKEN": "test-token"}, clear=True),
            patch(
                "lovely_assistant.services.tools._telegram_tools.httpx.AsyncClient"
            ) as mock_client_cls,
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            await respond_telegram("  Hello Elias  ")

            call_kwargs = mock_client.post.call_args
            assert call_kwargs[1]["json"]["text"] == "Hello Elias"


class TestRegisterTelegramTools:
    """Tests for the register_telegram_tools registration function."""

    def test_registers_respond_telegram(self):
        registry = ToolRegistry(ToolConfig())
        register_telegram_tools(registry)

        tool_names = registry.get_tool_names()
        assert "respond_telegram" in tool_names

    def test_registered_tool_count(self):
        registry = ToolRegistry(ToolConfig())
        initial = registry.backend_tool_count()
        register_telegram_tools(registry)
        assert registry.backend_tool_count() == initial + 1
