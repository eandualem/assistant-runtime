"""Tests for GET /api/media/video/{job_id} route."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.media import router
from assistant_runtime.services.media.models import VideoStatusResponse


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


class TestGetVideoStatus:
    """Tests for GET /api/media/video/{job_id}."""

    @pytest.mark.asyncio
    async def test_returns_200_with_status(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = VideoStatusResponse(
            job_id="vid-001",
            state="processing",
            video_url=None,
            error=None,
            provider="runway",
            model="runway:gen4-turbo",
        )
        response = await client.get("/api/media/video/vid-001")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "vid-001"
        assert data["state"] == "processing"
        assert data["provider"] == "runway"
        assert data["model"] == "runway:gen4-turbo"

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, client: AsyncClient, mock_media_service: MagicMock):
        mock_media_service.get_video_status.return_value = None
        response = await client.get("/api/media/video/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_not_found_detail_message(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = None
        response = await client.get("/api/media/video/nonexistent")
        data = response.json()
        assert data["detail"] == "Video job not found"

    @pytest.mark.asyncio
    async def test_calls_service_with_job_id(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = VideoStatusResponse(
            job_id="vid-xyz",
            state="submitted",
            video_url=None,
            error=None,
            provider="runway",
            model="runway:gen4-turbo",
        )
        await client.get("/api/media/video/vid-xyz")
        mock_media_service.get_video_status.assert_called_once_with("vid-xyz")

    @pytest.mark.asyncio
    async def test_completed_job_includes_video_url(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = VideoStatusResponse(
            job_id="vid-done",
            state="completed",
            video_url="https://cdn.runway.com/videos/vid-done.mp4",
            error=None,
            provider="runway",
            model="runway:gen4-turbo",
        )
        response = await client.get("/api/media/video/vid-done")
        data = response.json()
        assert data["state"] == "completed"
        assert data["video_url"] == "https://cdn.runway.com/videos/vid-done.mp4"
        assert data["error"] is None

    @pytest.mark.asyncio
    async def test_failed_job_includes_error(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = VideoStatusResponse(
            job_id="vid-fail",
            state="failed",
            video_url=None,
            error="Content moderation rejected the prompt",
            provider="luma",
            model="luma:ray-2",
        )
        response = await client.get("/api/media/video/vid-fail")
        data = response.json()
        assert data["state"] == "failed"
        assert data["error"] == "Content moderation rejected the prompt"
        assert data["video_url"] is None

    @pytest.mark.asyncio
    async def test_submitted_job_has_no_video_url_or_error(
        self, client: AsyncClient, mock_media_service: MagicMock
    ):
        mock_media_service.get_video_status.return_value = VideoStatusResponse(
            job_id="vid-new",
            state="submitted",
            video_url=None,
            error=None,
            provider="runway",
            model="runway:gen4-turbo",
        )
        response = await client.get("/api/media/video/vid-new")
        data = response.json()
        assert data["state"] == "submitted"
        assert data["video_url"] is None
        assert data["error"] is None
