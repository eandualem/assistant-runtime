"""Tests for GET /media/{image_id} route."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from lovely_assistant.app.routes.media import router


def _make_app(media_service: MagicMock) -> FastAPI:
    """Create a minimal FastAPI app with media routes and a mock service."""
    app = FastAPI()
    app.state.media_service = media_service
    app.include_router(router, prefix="/api")
    return app


@pytest.fixture
def mock_media_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
async def client(mock_media_service: MagicMock):
    """Async test client for media routes."""
    app = _make_app(mock_media_service)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


class TestGetImage:
    """Tests for GET /api/media/{image_id}."""

    @pytest.mark.asyncio
    async def test_returns_200_with_image_bytes(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_cached_image.return_value = (
            b"png-data",
            "image/png",
        )
        response = await client.get("/api/media/abc123")
        assert response.status_code == 200
        assert response.content == b"png-data"

    @pytest.mark.asyncio
    async def test_returns_correct_content_type(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_cached_image.return_value = (
            b"png-data",
            "image/png",
        )
        response = await client.get("/api/media/abc123")
        assert response.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_returns_cache_control_header(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_cached_image.return_value = (
            b"png-data",
            "image/png",
        )
        response = await client.get("/api/media/abc123")
        assert response.headers["cache-control"] == "public, max-age=3600"

    @pytest.mark.asyncio
    async def test_calls_service_with_image_id(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_cached_image.return_value = (
            b"data",
            "image/png",
        )
        await client.get("/api/media/my-image-id")
        mock_media_service.get_cached_image.assert_called_once_with("my-image-id")

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client: AsyncClient, mock_media_service: MagicMock):
        mock_media_service.get_cached_image.return_value = None
        response = await client.get("/api/media/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_not_found_detail_message(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_cached_image.return_value = None
        response = await client.get("/api/media/nonexistent")
        data = response.json()
        assert data["detail"] == "Image not found or expired"


class TestGetImageMimeTypes:
    """Verify correct content-type for different image formats."""

    @pytest.mark.asyncio
    async def test_jpeg_content_type(self, client: AsyncClient, mock_media_service: MagicMock):
        mock_media_service.get_cached_image.return_value = (
            b"\xff\xd8\xff\xe0",
            "image/jpeg",
        )
        response = await client.get("/api/media/jpeg-img")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.content == b"\xff\xd8\xff\xe0"

    @pytest.mark.asyncio
    async def test_webp_content_type(self, client: AsyncClient, mock_media_service: MagicMock):
        mock_media_service.get_cached_image.return_value = (
            b"RIFF\x00\x00\x00\x00WEBP",
            "image/webp",
        )
        response = await client.get("/api/media/webp-img")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/webp"
        assert response.content == b"RIFF\x00\x00\x00\x00WEBP"

    @pytest.mark.asyncio
    async def test_png_content_type(self, client: AsyncClient, mock_media_service: MagicMock):
        mock_media_service.get_cached_image.return_value = (
            b"\x89PNG\r\n\x1a\n",
            "image/png",
        )
        response = await client.get("/api/media/png-img")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content == b"\x89PNG\r\n\x1a\n"
