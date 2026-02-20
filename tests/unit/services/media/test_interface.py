"""Tests for MediaService lifecycle, generate_image, cache retrieval, and parsing."""

from unittest.mock import AsyncMock, patch

import pytest

from lovely_assistant.services.media.config import MediaConfig
from lovely_assistant.services.media.exceptions import MediaError, ProviderError
from lovely_assistant.services.media.interface import MediaService
from lovely_assistant.services.media.models import GeneratedImage, MediaResult

MODULE = "lovely_assistant.services.media.interface"


@pytest.fixture
def config():
    return MediaConfig()


@pytest.fixture
def service(config):
    return MediaService(config=config)


@pytest.fixture
async def started_service(service):
    await service.start()
    yield service
    await service.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    async def test_start_initializes_cache_and_started(self, service):
        await service.start()
        assert service._started is True
        assert service._cache is not None

    async def test_stop_clears_cache_and_unsets_started(self, service):
        await service.start()
        await service.stop()
        assert service._started is False
        assert service._cache is None

    async def test_health_check_before_start(self, service):
        health = await service.health_check()
        assert health == {"healthy": False}

    async def test_health_check_after_start(self, started_service):
        health = await started_service.health_check()
        assert health["healthy"] is True
        assert health["cached_images"] == 0
        assert health["default_model"] == "openai:gpt-image-1"

    async def test_health_check_after_stop(self, service):
        await service.start()
        await service.stop()
        health = await service.health_check()
        assert health == {"healthy": False}


# ---------------------------------------------------------------------------
# generate_image
# ---------------------------------------------------------------------------


class TestGenerateImage:
    """Tests for MediaService.generate_image."""

    def _make_generated_image(
        self,
        provider: str = "openai",
        model: str = "openai:gpt-image-1",
        mime_type: str = "image/png",
    ) -> GeneratedImage:
        return GeneratedImage(
            image_bytes=b"fake-image-bytes",
            mime_type=mime_type,
            provider=provider,
            model=model,
        )

    async def test_not_started_raises_media_error(self, service):
        with pytest.raises(MediaError, match="not started"):
            await service.generate_image(prompt="a cat")

    async def test_uses_default_model_from_config(self, started_service):
        generated = self._make_generated_image()
        with patch(
            f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated
        ) as mock_openai:
            await started_service.generate_image(prompt="a cat")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["model_name"] == "gpt-image-1"

    async def test_uses_default_size_from_config(self, started_service):
        generated = self._make_generated_image()
        with patch(
            f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated
        ) as mock_openai:
            await started_service.generate_image(prompt="a cat")
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["size"] == "1024x1024"

    async def test_uses_default_quality_from_config(self, started_service):
        generated = self._make_generated_image()
        with patch(
            f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated
        ) as mock_openai:
            await started_service.generate_image(prompt="a cat")
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["quality"] == "medium"

    async def test_overrides_model_size_quality(self, started_service):
        generated = self._make_generated_image(
            provider="google", model="google:imagen-4.0-generate-preview-06-06"
        )
        with patch(
            f"{MODULE}.generate_google", new_callable=AsyncMock, return_value=generated
        ) as mock_google:
            await started_service.generate_image(
                prompt="a cat",
                model="google:imagen-4.0-generate-preview-06-06",
                size="1536x1024",
                quality="high",
            )
            call_kwargs = mock_google.call_args.kwargs
            assert call_kwargs["model_name"] == "imagen-4.0-generate-preview-06-06"
            assert call_kwargs["size"] == "1536x1024"

    async def test_routes_to_openai_provider(self, started_service):
        generated = self._make_generated_image()
        with patch(
            f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated
        ) as mock_openai:
            await started_service.generate_image(prompt="a cat", model="openai:gpt-image-1")
            mock_openai.assert_called_once_with(
                prompt="a cat",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )

    async def test_routes_to_google_provider(self, started_service):
        generated = self._make_generated_image(
            provider="google", model="google:imagen-4.0-generate-preview-06-06"
        )
        with patch(
            f"{MODULE}.generate_google", new_callable=AsyncMock, return_value=generated
        ) as mock_google:
            await started_service.generate_image(
                prompt="a dog", model="google:imagen-4.0-generate-preview-06-06"
            )
            mock_google.assert_called_once_with(
                prompt="a dog",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )

    async def test_stores_result_in_cache(self, started_service):
        generated = self._make_generated_image()
        with patch(f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated):
            result = await started_service.generate_image(prompt="a cat")

        # Cache should now contain one entry
        assert len(started_service._cache) == 1

        # Retrieve via cache
        cached = started_service.get_cached_image(result.image_id)
        assert cached is not None
        image_bytes, mime_type = cached
        assert image_bytes == b"fake-image-bytes"
        assert mime_type == "image/png"

    async def test_returns_media_result_with_correct_fields(self, started_service):
        generated = self._make_generated_image()
        with patch(f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated):
            result = await started_service.generate_image(prompt="a cat")

        assert isinstance(result, MediaResult)
        assert result.provider == "openai"
        assert result.model == "openai:gpt-image-1"
        assert result.mime_type == "image/png"
        assert result.image_id  # non-empty

    async def test_url_matches_api_media_format(self, started_service):
        generated = self._make_generated_image()
        with patch(f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated):
            result = await started_service.generate_image(prompt="a cat")

        assert result.url == f"/api/media/{result.image_id}"


# ---------------------------------------------------------------------------
# get_cached_image
# ---------------------------------------------------------------------------


class TestGetCachedImage:
    def test_returns_none_when_not_started(self, service):
        assert service.get_cached_image("some-id") is None

    async def test_returns_none_for_missing_image_id(self, started_service):
        assert started_service.get_cached_image("nonexistent-id") is None

    async def test_returns_bytes_and_mime_type_for_cached_image(self, started_service):
        generated = GeneratedImage(
            image_bytes=b"test-png-data",
            mime_type="image/png",
            provider="openai",
            model="openai:gpt-image-1",
        )
        with patch(f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated):
            result = await started_service.generate_image(prompt="test")

        cached = started_service.get_cached_image(result.image_id)
        assert cached is not None
        image_bytes, mime_type = cached
        assert image_bytes == b"test-png-data"
        assert mime_type == "image/png"


# ---------------------------------------------------------------------------
# _parse_model_id
# ---------------------------------------------------------------------------


class TestParseModelId:
    def test_openai_model(self):
        provider, model_name = MediaService._parse_model_id("openai:gpt-image-1")
        assert provider == "openai"
        assert model_name == "gpt-image-1"

    def test_google_model(self):
        provider, model_name = MediaService._parse_model_id(
            "google:imagen-4.0-generate-preview-06-06"
        )
        assert provider == "google"
        assert model_name == "imagen-4.0-generate-preview-06-06"

    def test_no_colon_raises_provider_error(self):
        with pytest.raises(ProviderError, match="Invalid model ID"):
            MediaService._parse_model_id("gpt-image-1")

    def test_unknown_provider_raises_provider_error(self):
        with pytest.raises(ProviderError, match="Unknown image provider"):
            MediaService._parse_model_id("midjourney:v6")

    def test_colon_in_model_name_preserves_full_name(self):
        provider, model_name = MediaService._parse_model_id("openai:model:variant")
        assert provider == "openai"
        assert model_name == "model:variant"


# ---------------------------------------------------------------------------
# _call_provider
# ---------------------------------------------------------------------------


class TestCallProvider:
    async def test_openai_calls_generate_openai(self):
        generated = GeneratedImage(
            image_bytes=b"data",
            mime_type="image/png",
            provider="openai",
            model="openai:gpt-image-1",
        )
        with patch(
            f"{MODULE}.generate_openai", new_callable=AsyncMock, return_value=generated
        ) as mock_openai:
            result = await MediaService._call_provider(
                provider="openai",
                model_name="gpt-image-1",
                prompt="a bird",
                size="1024x1024",
                quality="medium",
            )
            mock_openai.assert_called_once_with(
                prompt="a bird",
                model_name="gpt-image-1",
                size="1024x1024",
                quality="medium",
            )
            assert result == generated

    async def test_google_calls_generate_google(self):
        generated = GeneratedImage(
            image_bytes=b"data",
            mime_type="image/png",
            provider="google",
            model="google:imagen-4.0-generate-preview-06-06",
        )
        with patch(
            f"{MODULE}.generate_google", new_callable=AsyncMock, return_value=generated
        ) as mock_google:
            result = await MediaService._call_provider(
                provider="google",
                model_name="imagen-4.0-generate-preview-06-06",
                prompt="a bird",
                size="1024x1024",
                quality="medium",
            )
            mock_google.assert_called_once_with(
                prompt="a bird",
                model_name="imagen-4.0-generate-preview-06-06",
                size="1024x1024",
            )
            assert result == generated

    async def test_unknown_provider_raises_provider_error(self):
        with pytest.raises(ProviderError, match="No handler for provider"):
            await MediaService._call_provider(
                provider="midjourney",
                model_name="v6",
                prompt="a bird",
                size="1024x1024",
                quality="medium",
            )
