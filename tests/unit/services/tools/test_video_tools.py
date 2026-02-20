"""Tests for video generation tools — registration and handler behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lovely_assistant.services.media.exceptions import ContentPolicyError, MediaError, ProviderError
from lovely_assistant.services.media.models import VideoResult
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools._video_tools import register_video_tools
from lovely_assistant.services.tools.config import ToolConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry():
    return ToolRegistry(ToolConfig())


@pytest.fixture
def mock_media_service():
    service = MagicMock()
    service.generate_video = AsyncMock()
    return service


@pytest.fixture
def registered_registry(registry, mock_media_service):
    """Registry with video tools already registered."""
    register_video_tools(registry, mock_media_service)
    return registry


@pytest.fixture
def generate_video_handler(registered_registry):
    """Extract the generate_video handler closure from the registry."""
    return registered_registry._backend_handlers["generate_video"]


@pytest.fixture
def sample_video_result():
    """A valid VideoResult for a successfully submitted generation job."""
    return VideoResult(
        job_id="vid-abc123",
        status="submitted",
        provider="runway",
        model="runway:gen4-turbo",
        status_url="/api/media/video/vid-abc123",
    )


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registers_generate_video(self, registered_registry):
        assert "generate_video" in registered_registry.get_tool_names()

    def test_registered_as_backend(self, registered_registry):
        defn = registered_registry._backend_definitions["generate_video"]
        assert defn.category == "backend"

    def test_has_description(self, registered_registry):
        defn = registered_registry._backend_definitions["generate_video"]
        assert defn.description
        assert "video" in defn.description.lower()

    def test_has_handler(self, registered_registry):
        assert callable(registered_registry._backend_handlers["generate_video"])

    def test_prompt_is_required_parameter(self, registered_registry):
        schema = registered_registry._backend_definitions["generate_video"].parameters_schema
        assert "prompt" in schema["properties"]
        assert "prompt" in schema["required"]

    def test_optional_parameters_present(self, registered_registry):
        schema = registered_registry._backend_definitions["generate_video"].parameters_schema
        assert "model" in schema["properties"]
        assert "duration" in schema["properties"]

    def test_registers_exactly_one_tool(self, registry, mock_media_service):
        register_video_tools(registry, mock_media_service)
        assert len(registry._backend_definitions) == 1


# ---------------------------------------------------------------------------
# TestGenerateVideoSuccess
# ---------------------------------------------------------------------------


class TestGenerateVideoSuccess:
    async def test_successful_generation(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        result = await generate_video_handler(prompt="A drone flying over mountains")

        assert result["success"] is True
        assert result["job_id"] == "vid-abc123"
        assert result["status"] == "submitted"
        assert result["status_url"] == "/api/media/video/vid-abc123"
        assert result["provider"] == "runway"
        assert result["model"] == "runway:gen4-turbo"
        assert "message" in result

    async def test_passes_prompt_to_service(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        await generate_video_handler(prompt="A cat playing piano")

        mock_media_service.generate_video.assert_awaited_once()
        call_kwargs = mock_media_service.generate_video.call_args.kwargs
        assert call_kwargs["prompt"] == "A cat playing piano"

    async def test_passes_explicit_model(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        await generate_video_handler(prompt="test", model="luma:ray-2")

        call_kwargs = mock_media_service.generate_video.call_args.kwargs
        assert call_kwargs["model"] == "luma:ray-2"

    async def test_passes_duration_to_service(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        await generate_video_handler(prompt="test", duration=10)

        call_kwargs = mock_media_service.generate_video.call_args.kwargs
        assert call_kwargs["duration"] == 10

    async def test_default_duration_is_5(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        await generate_video_handler(prompt="test")

        call_kwargs = mock_media_service.generate_video.call_args.kwargs
        assert call_kwargs["duration"] == 5


# ---------------------------------------------------------------------------
# TestEmptyStringConversion
# ---------------------------------------------------------------------------


class TestEmptyStringConversion:
    """Empty string model parameter should be converted to None before calling the service."""

    async def test_empty_model_becomes_none(
        self, generate_video_handler, mock_media_service, sample_video_result
    ):
        mock_media_service.generate_video.return_value = sample_video_result

        await generate_video_handler(prompt="test", model="")

        call_kwargs = mock_media_service.generate_video.call_args.kwargs
        assert call_kwargs["model"] is None


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_content_policy_error(self, generate_video_handler, mock_media_service):
        mock_media_service.generate_video.side_effect = ContentPolicyError(
            "Prompt violates safety guidelines"
        )

        result = await generate_video_handler(prompt="bad prompt")

        assert result["success"] is False
        assert "Content policy violation" in result["error"]
        assert "safety guidelines" in result["error"]

    async def test_media_error(self, generate_video_handler, mock_media_service):
        mock_media_service.generate_video.side_effect = MediaError("Service not started")

        result = await generate_video_handler(prompt="test")

        assert result["success"] is False
        assert "Service not started" in result["error"]

    async def test_provider_error_caught_by_media_error_handler(
        self, generate_video_handler, mock_media_service
    ):
        """ProviderError is a subclass of MediaError and should be caught by the MediaError handler."""
        mock_media_service.generate_video.side_effect = ProviderError("Runway API returned 500")

        result = await generate_video_handler(prompt="test")

        assert result["success"] is False
        assert "Runway API returned 500" in result["error"]

    async def test_error_results_have_no_success_keys(
        self, generate_video_handler, mock_media_service
    ):
        """Error results should only have 'success' and 'error' — no video data."""
        mock_media_service.generate_video.side_effect = MediaError("fail")

        result = await generate_video_handler(prompt="test")

        assert "job_id" not in result
        assert "status_url" not in result
        assert "status" not in result
        assert "provider" not in result
        assert "model" not in result
        assert "message" not in result
