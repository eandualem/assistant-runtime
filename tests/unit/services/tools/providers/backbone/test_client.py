"""Tests for the shared backbone HTTP client."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_runtime.services.tools.providers.backbone._client import backbone_request

MODULE = "assistant_runtime.services.tools.providers.backbone._client"


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


class TestCredentialsOverCleartext:
    async def test_bearer_sent_over_https_and_to_localhost(self, monkeypatch):
        from assistant_runtime.services.tools.providers.backbone._client import (
            _credentials_allowed,
        )

        assert _credentials_allowed("https://backbone.example.com") is True
        assert _credentials_allowed("http://127.0.0.1:7120") is True
        assert _credentials_allowed("http://localhost:7120") is True
        assert _credentials_allowed("http://backbone.internal:7120") is False

    @pytest.mark.parametrize("url", ["https://backbone.example.com", "http://127.0.0.1:7120"])
    async def test_key_is_sent_over_tls_or_to_localhost(self, monkeypatch, url):
        monkeypatch.setenv("BACKBONE_URL", url)
        monkeypatch.setenv("BACKBONE_API_KEY", "secret")
        captured = {}

        class _Response:
            status_code = 200

            def json(self):
                return {"ok": True}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def request(self, method, url, headers=None, json=None, params=None):
                captured["headers"] = headers
                return _Response()

        monkeypatch.setattr(
            "assistant_runtime.services.tools.providers.backbone._client.httpx.AsyncClient", _Client
        )
        await backbone_request("GET", "/api/agents")
        assert captured["headers"]["Authorization"] == "Bearer secret"

    async def test_key_is_withheld_from_a_remote_cleartext_url(self, monkeypatch):
        monkeypatch.setenv("BACKBONE_URL", "http://backbone.internal:7120")
        monkeypatch.setenv("BACKBONE_API_KEY", "secret")
        captured = {}

        class _Response:
            status_code = 200

            def json(self):
                return {"ok": True}

        class _Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

            async def request(self, method, url, headers=None, json=None, params=None):
                captured["headers"] = headers
                return _Response()

        monkeypatch.setattr(
            "assistant_runtime.services.tools.providers.backbone._client.httpx.AsyncClient", _Client
        )
        status, _ = await backbone_request("GET", "/api/agents")
        assert status == 200
        assert "Authorization" not in captured["headers"]


class TestPayloadShapes:
    def test_error_helpers_tolerate_non_objects(self):
        from assistant_runtime.services.tools.providers.backbone._client import (
            backbone_detail,
            backbone_error,
        )

        assert backbone_error([]) == "Request failed"
        assert backbone_error(None) == "Request failed"
        assert backbone_error({"message": "nope"}) == "nope"
        assert backbone_detail("oops") == "Unknown error"
        assert backbone_detail({"detail": "bad"}) == "bad"
