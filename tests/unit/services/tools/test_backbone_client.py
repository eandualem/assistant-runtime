"""Tests for the shared backbone HTTP client."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

from lovely_assistant.services.tools._backbone_client import backbone_request

MODULE = "lovely_assistant.services.tools._backbone_client"


class TestBackboneRequest:
    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_successful_get(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": [], "total": 0}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == 200
        assert data == {"items": [], "total": 0}

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_successful_post(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "room-1", "topic": "test"}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("POST", "/api/rooms", json_body={"topic": "test"})
        assert status == 201
        assert data["id"] == "room-1"

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_network_error(self, mock_client_cls):
        import httpx

        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.HTTPError("Connection refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == -1
        assert "HTTP error" in data["error"]
        assert data["error_code"] == "BACKBONE_HTTP_ERROR"

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_timeout(self, mock_client_cls):
        import httpx

        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == -1
        assert "timed out" in data["error"]
        assert data["error_code"] == "BACKBONE_TIMEOUT"

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_auth_header_set(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await backbone_request("GET", "/api/rooms")

        call_kwargs = mock_client.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer test-key"

    @patch.dict(os.environ, {"BACKBONE_API_KEY": ""})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_no_auth_when_key_empty(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await backbone_request("GET", "/api/rooms")

        call_kwargs = mock_client.request.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert "Authorization" not in headers

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_params_passed(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await backbone_request("GET", "/api/rooms", params={"state": "active"})

        call_kwargs = mock_client.request.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        assert params == {"state": "active"}


class TestBackboneRetry:
    """Tests for retry behavior in backbone_request."""

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_retries_on_timeout_then_succeeds(self, mock_client_cls):
        """backbone_request retries on TimeoutException and succeeds."""
        import httpx

        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}

        async def _request_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.TimeoutException("timed out")
            return mock_response

        mock_client = AsyncMock()
        mock_client.request = _request_side_effect
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == 200
        assert data == {"ok": True}

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_retries_on_connect_error_then_succeeds(self, mock_client_cls):
        """backbone_request retries on ConnectError and succeeds."""
        import httpx

        call_count = 0
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"recovered": True}

        async def _request_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("connection refused")
            return mock_response

        mock_client = AsyncMock()
        mock_client.request = _request_side_effect
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == 200
        assert data == {"recovered": True}

    @patch.dict(os.environ, {"BACKBONE_API_KEY": "test-key"})
    @patch(f"{MODULE}.httpx.AsyncClient")
    async def test_gives_up_after_max_retries(self, mock_client_cls):
        """backbone_request returns error tuple after exhausting retries."""
        import httpx

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("permanent timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        status, data = await backbone_request("GET", "/api/rooms")
        assert status == -1
        assert "timed out" in data["error"]
        assert data["error_code"] == "BACKBONE_TIMEOUT"
