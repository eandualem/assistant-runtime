"""Tests for media generation tools — registration and handler behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lovely_assistant.services.media.exceptions import ContentPolicyError, MediaError, ProviderError
from lovely_assistant.services.media.models import MediaResult
from lovely_assistant.services.tools._media_tools import register_media_tools
from lovely_assistant.services.tools._registry import ToolRegistry
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
    service.generate_image = AsyncMock()
    return service


@pytest.fixture
def registered_registry(registry, mock_media_service):
    """Registry with media tools already registered."""
    register_media_tools(registry, mock_media_service)
    return registry


@pytest.fixture
def generate_image_handler(registered_registry):
    """Extract the generate_image handler closure from the registry."""
    return registered_registry._backend_handlers["generate_image"]


@pytest.fixture
def sample_media_result():
    """A valid MediaResult for successful generation."""
    return MediaResult(
        image_id="img-abc123",
        url="/api/media/img-abc123",
        provider="openai",
        model="openai:gpt-image-1",
        mime_type="image/png",
    )


# ---------------------------------------------------------------------------
# TestRegistration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_registers_generate_image(self, registered_registry):
        assert "generate_image" in registered_registry.get_tool_names()

    def test_registered_as_backend(self, registered_registry):
        defn = registered_registry._backend_definitions["generate_image"]
        assert defn.category == "backend"

    def test_has_description(self, registered_registry):
        defn = registered_registry._backend_definitions["generate_image"]
        assert defn.description
        assert "image" in defn.description.lower()

    def test_has_handler(self, registered_registry):
        assert callable(registered_registry._backend_handlers["generate_image"])

    def test_prompt_is_required_parameter(self, registered_registry):
        schema = registered_registry._backend_definitions["generate_image"].parameters_schema
        assert "prompt" in schema["properties"]
        assert "prompt" in schema["required"]

    def test_optional_parameters_present(self, registered_registry):
        schema = registered_registry._backend_definitions["generate_image"].parameters_schema
        assert "model" in schema["properties"]
        assert "size" in schema["properties"]
        assert "quality" in schema["properties"]

    def test_optional_parameters_have_defaults(self, registered_registry):
        schema = registered_registry._backend_definitions["generate_image"].parameters_schema
        props = schema["properties"]
        assert props["model"]["default"] == ""
        assert props["size"]["default"] == ""
        assert props["quality"]["default"] == ""

    def test_registers_exactly_one_tool(self, registry, mock_media_service):
        register_media_tools(registry, mock_media_service)
        assert len(registry._backend_definitions) == 1


# ---------------------------------------------------------------------------
# TestGenerateImageSuccess
# ---------------------------------------------------------------------------


class TestGenerateImageSuccess:
    async def test_successful_generation(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        result = await generate_image_handler(prompt="A sunset over mountains")

        assert result["success"] is True
        assert result["image_url"] == "/api/media/img-abc123"
        assert result["image_id"] == "img-abc123"
        assert result["model"] == "openai:gpt-image-1"
        assert result["provider"] == "openai"

    async def test_passes_prompt_to_service(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="A cat wearing a hat")

        mock_media_service.generate_image.assert_awaited_once()
        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["prompt"] == "A cat wearing a hat"

    async def test_passes_explicit_model(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", model="google:imagen-4.0")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["model"] == "google:imagen-4.0"

    async def test_passes_explicit_size(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", size="1536x1024")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["size"] == "1536x1024"

    async def test_passes_explicit_quality(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", quality="high")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["quality"] == "high"


# ---------------------------------------------------------------------------
# TestEmptyStringConversion
# ---------------------------------------------------------------------------


class TestEmptyStringConversion:
    """Empty string parameters should be converted to None before calling the service."""

    async def test_empty_model_becomes_none(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", model="")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["model"] is None

    async def test_empty_size_becomes_none(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", size="")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["size"] is None

    async def test_empty_quality_becomes_none(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test", quality="")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["quality"] is None

    async def test_all_defaults_become_none(
        self, generate_image_handler, mock_media_service, sample_media_result
    ):
        """When called with only the prompt, all optional params default to '' -> None."""
        mock_media_service.generate_image.return_value = sample_media_result

        await generate_image_handler(prompt="test")

        call_kwargs = mock_media_service.generate_image.call_args.kwargs
        assert call_kwargs["model"] is None
        assert call_kwargs["size"] is None
        assert call_kwargs["quality"] is None


# ---------------------------------------------------------------------------
# TestErrorHandling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_content_policy_error(self, generate_image_handler, mock_media_service):
        mock_media_service.generate_image.side_effect = ContentPolicyError(
            "Prompt violates safety guidelines"
        )

        result = await generate_image_handler(prompt="bad prompt")

        assert result["success"] is False
        assert "Content policy violation" in result["error"]
        assert "safety guidelines" in result["error"]

    async def test_media_error(self, generate_image_handler, mock_media_service):
        mock_media_service.generate_image.side_effect = MediaError("Service not started")

        result = await generate_image_handler(prompt="test")

        assert result["success"] is False
        assert "Service not started" in result["error"]

    async def test_provider_error_caught_by_media_error_handler(
        self, generate_image_handler, mock_media_service
    ):
        """ProviderError is a subclass of MediaError and should be caught by the MediaError handler."""
        mock_media_service.generate_image.side_effect = ProviderError("OpenAI API returned 500")

        result = await generate_image_handler(prompt="test")

        assert result["success"] is False
        assert "OpenAI API returned 500" in result["error"]

    async def test_content_policy_error_prefix_format(
        self, generate_image_handler, mock_media_service
    ):
        """ContentPolicyError results should have the 'Content policy violation: ' prefix."""
        mock_media_service.generate_image.side_effect = ContentPolicyError("blocked")

        result = await generate_image_handler(prompt="test")

        assert result["error"].startswith("Content policy violation:")

    async def test_media_error_no_prefix(self, generate_image_handler, mock_media_service):
        """MediaError results should NOT have the 'Content policy violation' prefix."""
        mock_media_service.generate_image.side_effect = MediaError("generic failure")

        result = await generate_image_handler(prompt="test")

        assert "Content policy violation" not in result["error"]

    async def test_error_results_have_no_image_keys(
        self, generate_image_handler, mock_media_service
    ):
        """Error results should only have 'success' and 'error' — no image data."""
        mock_media_service.generate_image.side_effect = MediaError("fail")

        result = await generate_image_handler(prompt="test")

        assert "image_url" not in result
        assert "image_id" not in result
        assert "model" not in result
        assert "provider" not in result
