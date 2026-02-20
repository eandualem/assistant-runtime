"""Tests for image generation provider functions (OpenAI and Google)."""

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lovely_assistant.services.media._providers import (
    _SIZE_TO_ASPECT_RATIO,
    generate_google,
    generate_openai,
)
from lovely_assistant.services.media.exceptions import (
    ContentPolicyError,
    ProviderError,
    ProviderNotConfiguredError,
)
from lovely_assistant.services.media.models import GeneratedImage

MODULE = "lovely_assistant.services.media._providers"

# Shared test data
FAKE_PNG = b"fake-png-data"
FAKE_B64 = base64.b64encode(FAKE_PNG).decode()


class TestGenerateOpenAI:
    """Tests for the generate_openai provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="OPENAI_API_KEY"):
            await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_successful_generation(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_image_item = SimpleNamespace(b64_json=FAKE_B64)
        mock_response = SimpleNamespace(data=[mock_image_item])

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

        assert isinstance(result, GeneratedImage)
        assert result.image_bytes == FAKE_PNG
        assert result.mime_type == "image/png"
        assert result.provider == "openai"
        assert result.model == "openai:gpt-image-1"

    async def test_returns_correct_model_field(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_image_item = SimpleNamespace(b64_json=FAKE_B64)
        mock_response = SimpleNamespace(data=[mock_image_item])

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await generate_openai(
                prompt="test",
                model_name="dall-e-3",
                size="1024x1024",
                quality="high",
            )

        assert result.model == "openai:dall-e-3"

    async def test_empty_image_data_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_image_item = SimpleNamespace(b64_json=None)
        mock_response = SimpleNamespace(data=[mock_image_item])

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ProviderError, match="empty image data"),
        ):
            await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_bad_request_content_policy_raises_content_policy_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from openai import BadRequestError

        error = BadRequestError(
            message="Your request was rejected as a result of content_policy violation",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(side_effect=error)

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ContentPolicyError, match="content policy rejection"),
        ):
            await generate_openai(
                prompt="bad content",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_bad_request_safety_raises_content_policy_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from openai import BadRequestError

        error = BadRequestError(
            message="Request rejected due to safety system",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(side_effect=error)

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ContentPolicyError, match="content policy rejection"),
        ):
            await generate_openai(
                prompt="bad content",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_bad_request_without_policy_keywords_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from openai import BadRequestError

        error = BadRequestError(
            message="Invalid image size parameter",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        )

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(side_effect=error)

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ProviderError, match="OpenAI API error"),
        ):
            await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="invalid",
                quality="medium",
            )

    async def test_generic_openai_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        from openai import OpenAIError

        error = OpenAIError("Connection timeout")

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(side_effect=error)

        with (
            patch("openai.AsyncOpenAI", return_value=mock_client),
            pytest.raises(ProviderError, match="OpenAI API error"),
        ):
            await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_api_key_passed_to_client(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-my-specific-key")

        mock_image_item = SimpleNamespace(b64_json=FAKE_B64)
        mock_response = SimpleNamespace(data=[mock_image_item])

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client) as mock_cls:
            await generate_openai(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

            mock_cls.assert_called_once_with(api_key="sk-my-specific-key")

    async def test_generate_call_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        mock_image_item = SimpleNamespace(b64_json=FAKE_B64)
        mock_response = SimpleNamespace(data=[mock_image_item])

        mock_client = MagicMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            await generate_openai(
                prompt="a beautiful sunset",
                model_name="gpt-image-1",
                size="1536x1024",
                quality="high",
            )

            mock_client.images.generate.assert_called_once_with(
                prompt="a beautiful sunset",
                model="gpt-image-1",
                size="1536x1024",
                quality="high",
                response_format="b64_json",
                n=1,
            )


class TestGenerateGoogle:
    """Tests for the generate_google provider function."""

    async def test_missing_api_key_raises_not_configured(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with pytest.raises(ProviderNotConfiguredError, match="GOOGLE_API_KEY"):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_successful_generation(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

        fake_image_bytes = b"fake-google-png"
        mock_image_data = SimpleNamespace(
            image_bytes=fake_image_bytes,
            mime_type="image/png",
        )
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
        ):
            result = await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

        assert isinstance(result, GeneratedImage)
        assert result.image_bytes == fake_image_bytes
        assert result.mime_type == "image/png"
        assert result.provider == "google"
        assert result.model == "google:imagen-4.0-generate-preview-06-06"

    async def test_returns_correct_model_field(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        fake_bytes = b"img-data"
        mock_image_data = SimpleNamespace(image_bytes=fake_bytes, mime_type="image/png")
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
        ):
            result = await generate_google(
                prompt="test",
                model_name="custom-model",
                size="1024x1024",
            )

        assert result.model == "google:custom-model"

    async def test_no_images_returned_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_response = SimpleNamespace(generated_images=[])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="no images"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_none_generated_images_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_response = SimpleNamespace(generated_images=None)

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="no images"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_empty_image_data_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_image_data = SimpleNamespace(image_bytes=None, mime_type=None)
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="empty image data"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_none_image_object_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_generated = SimpleNamespace(image=None)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="empty image data"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_safety_error_raises_content_policy_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(
            side_effect=Exception("Image generation failed due to safety filters"),
        )

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ContentPolicyError, match="content policy rejection"),
        ):
            await generate_google(
                prompt="bad content",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_blocked_error_raises_content_policy_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(
            side_effect=Exception("Request was blocked by the API"),
        )

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ContentPolicyError, match="content policy rejection"),
        ):
            await generate_google(
                prompt="bad content",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_generic_error_raises_provider_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(
            side_effect=Exception("Network timeout"),
        )

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="Google API error"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_provider_error_is_not_rewrapped(self, monkeypatch):
        """ProviderError raised inside the try block is re-raised as-is."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_response = SimpleNamespace(generated_images=[])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
            pytest.raises(ProviderError, match="no images"),
        ):
            await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_mime_type_from_response(self, monkeypatch):
        """When the image object has a mime_type attribute, it is used."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        mock_image_data = SimpleNamespace(image_bytes=b"jpeg-data", mime_type="image/jpeg")
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
        ):
            result = await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

        assert result.mime_type == "image/jpeg"

    async def test_mime_type_defaults_to_png(self, monkeypatch):
        """When mime_type is absent from the image object, defaults to image/png."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        # Use a MagicMock without mime_type attribute to trigger getattr fallback
        mock_image_data = MagicMock(spec=[])
        mock_image_data.image_bytes = b"some-data"
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig"),
        ):
            result = await generate_google(
                prompt="a cat",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

        assert result.mime_type == "image/png"


class TestSizeToAspectRatioMapping:
    """Tests for the _SIZE_TO_ASPECT_RATIO mapping used by generate_google."""

    def test_square_mapping(self):
        assert _SIZE_TO_ASPECT_RATIO["1024x1024"] == "1:1"

    def test_landscape_mapping(self):
        assert _SIZE_TO_ASPECT_RATIO["1536x1024"] == "3:2"

    def test_portrait_mapping(self):
        assert _SIZE_TO_ASPECT_RATIO["1024x1536"] == "2:3"

    def test_unknown_size_defaults_to_square(self):
        """Unknown sizes fall back to 1:1 via dict.get default."""
        assert _SIZE_TO_ASPECT_RATIO.get("800x600", "1:1") == "1:1"
        assert _SIZE_TO_ASPECT_RATIO.get("unknown", "1:1") == "1:1"

    async def test_aspect_ratio_passed_to_google_client(self, monkeypatch):
        """Verify that generate_google passes the mapped aspect ratio to the config."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        fake_bytes = b"img-data"
        mock_image_data = SimpleNamespace(image_bytes=fake_bytes, mime_type="image/png")
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        mock_config_cls = MagicMock()

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig", mock_config_cls),
        ):
            await generate_google(
                prompt="test",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1536x1024",
            )

            # Verify GenerateImagesConfig was called with the correct aspect ratio
            mock_config_cls.assert_called_once_with(
                number_of_images=1,
                aspect_ratio="3:2",
            )

    async def test_unknown_size_uses_square_aspect_ratio(self, monkeypatch):
        """When an unknown size is passed, the default 1:1 aspect ratio is used."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        fake_bytes = b"img-data"
        mock_image_data = SimpleNamespace(image_bytes=fake_bytes, mime_type="image/png")
        mock_generated = SimpleNamespace(image=mock_image_data)
        mock_response = SimpleNamespace(generated_images=[mock_generated])

        mock_client = MagicMock()
        mock_client.aio.models.generate_images = AsyncMock(return_value=mock_response)

        mock_config_cls = MagicMock()

        with (
            patch("google.genai.Client", return_value=mock_client),
            patch("google.genai.types.GenerateImagesConfig", mock_config_cls),
        ):
            await generate_google(
                prompt="test",
                model_name="imagen-4.0-generate-preview-06-06",
                size="800x600",
            )

            mock_config_cls.assert_called_once_with(
                number_of_images=1,
                aspect_ratio="1:1",
            )
