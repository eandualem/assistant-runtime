"""MediaService — image and video generation facade with lifecycle management."""

from __future__ import annotations

from loguru import logger

from lovely_assistant.services.media._cache import ImageCache
from lovely_assistant.services.media._job_tracker import JobTracker
from lovely_assistant.services.media._providers import generate_google, generate_openai
from lovely_assistant.services.media._video_providers import (
    poll_luma,
    poll_runway,
    submit_luma,
    submit_runway,
)
from lovely_assistant.services.media.config import MediaConfig
from lovely_assistant.services.media.exceptions import MediaError, ProviderError
from lovely_assistant.services.media.models import (
    GeneratedImage,
    MediaResult,
    VideoResult,
    VideoStatusResponse,
)

# Provider prefix → generator function (images)
_PROVIDER_GENERATORS: dict[str, object] = {
    "openai": generate_openai,
    "google": generate_google,
}

# Provider prefix → (submit_fn, poll_fn) for video
_VIDEO_PROVIDERS: dict[str, tuple] = {
    "runway": (submit_runway, poll_runway),
    "luma": (submit_luma, poll_luma),
}


class MediaService:
    """Image and video generation facade. Implements LifecycleAware."""

    def __init__(self, config: MediaConfig) -> None:
        self._config = config
        self._cache: ImageCache | None = None
        self._job_tracker: JobTracker | None = None
        self._runtime_settings = None
        self._started = False

    async def start(self) -> None:
        """Initialize the image cache and video job tracker."""
        self._cache = ImageCache(
            ttl_seconds=self._config.cache_ttl_seconds,
            max_items=self._config.cache_max_items,
        )
        self._job_tracker = JobTracker(
            max_concurrent=self._config.video_max_concurrent_jobs,
            poll_interval=self._config.video_poll_interval_seconds,
            timeout=self._config.video_timeout_seconds,
        )
        self._started = True
        logger.info(
            "Media service started",
            default_model=self._config.default_image_model,
            default_video_model=self._config.default_video_model,
            cache_ttl=self._config.cache_ttl_seconds,
            cache_max=self._config.cache_max_items,
        )

    async def stop(self) -> None:
        """Clear cache, cancel video jobs, and shut down."""
        if self._job_tracker is not None:
            await self._job_tracker.stop_all()
        self._job_tracker = None
        self._cache = None
        self._started = False
        logger.info("Media service stopped")

    async def health_check(self) -> dict:
        """Report health status."""
        if not self._started or self._cache is None:
            return {"healthy": False}
        result: dict = {
            "healthy": True,
            "cached_images": len(self._cache),
            "default_model": self._config.default_image_model,
        }
        if self._job_tracker is not None:
            result["active_video_jobs"] = self._job_tracker.active_count()
        return result

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: str | None = None,
        quality: str | None = None,
    ) -> MediaResult:
        """Generate an image, cache it, and return a serving URL.

        Args:
            prompt: Text description of the image.
            model: Full model ID with provider prefix (e.g. "openai:gpt-image-1").
                   Defaults to config.default_image_model.
            size: Image size (e.g. "1024x1024"). Defaults to config.default_size.
            quality: Quality level (e.g. "medium", "high"). Defaults to config.default_quality.

        Returns:
            MediaResult with image_id and serving URL.

        Raises:
            MediaError: If the service is not started.
            ProviderError: If the provider API call fails.
            ProviderNotConfiguredError: If the API key is missing.
            ContentPolicyError: If the prompt violates content policy.
        """
        self._ensure_started()

        config_default = self._config.default_image_model
        runtime_default = (
            self._runtime_settings.get("default_image_model", config_default)
            if self._runtime_settings
            else config_default
        )
        resolved_model = model or runtime_default
        resolved_size = size or self._config.default_size
        resolved_quality = quality or self._config.default_quality

        # Parse provider prefix
        provider, model_name = self._parse_model_id(resolved_model)

        # Route to provider
        generated = await self._call_provider(
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            size=resolved_size,
            quality=resolved_quality,
        )

        # Cache the result
        image_id = self._cache.store(generated.image_bytes, generated.mime_type)

        result = MediaResult(
            image_id=image_id,
            url=f"/api/media/{image_id}",
            provider=generated.provider,
            model=generated.model,
            mime_type=generated.mime_type,
        )
        logger.info(
            "Image generated and cached",
            image_id=image_id,
            provider=provider,
            model=resolved_model,
        )
        return result

    def get_cached_image(self, image_id: str) -> tuple[bytes, str] | None:
        """Retrieve a cached image by ID.

        Returns:
            (image_bytes, mime_type) or None if not found/expired.
        """
        if self._cache is None:
            return None
        return self._cache.get(image_id)

    async def generate_video(
        self,
        prompt: str,
        model: str | None = None,
        duration: int = 5,
    ) -> VideoResult:
        """Submit a video generation job.

        Args:
            prompt: Text description of the video.
            model: Full model ID with provider prefix (e.g. "runway:gen4-turbo").
                   Defaults to config.default_video_model.
            duration: Video duration in seconds.

        Returns:
            VideoResult with job_id and status_url for polling.

        Raises:
            MediaError: If the service is not started.
            ProviderError: If the provider API call fails.
            ProviderNotConfiguredError: If the API key is missing.
        """
        self._ensure_started()

        config_default = self._config.default_video_model
        runtime_default = (
            self._runtime_settings.get("default_video_model", config_default)
            if self._runtime_settings
            else config_default
        )
        resolved_model = model or runtime_default
        provider, model_name = self._parse_video_model_id(resolved_model)

        # Submit to provider and get poll function
        provider_job_id = await self._submit_video(provider, model_name, prompt, duration)
        video_poll_fn = self._get_video_poll_fn(provider)

        # Register with job tracker
        job = self._job_tracker.submit(
            provider_job_id=provider_job_id,
            provider=provider,
            model=resolved_model,
            poll_fn=video_poll_fn,
        )

        result = VideoResult(
            job_id=job.internal_id,
            status="submitted",
            provider=provider,
            model=resolved_model,
            status_url=f"/api/media/video/{job.internal_id}",
        )
        logger.info(
            "Video generation submitted",
            job_id=job.internal_id,
            provider=provider,
            model=resolved_model,
        )
        return result

    def get_video_status(self, job_id: str) -> VideoStatusResponse | None:
        """Get the current status of a video generation job.

        Returns None if the job_id is not found.
        """
        if self._job_tracker is None:
            return None
        job = self._job_tracker.get_status(job_id)
        if job is None:
            return None
        return VideoStatusResponse(
            job_id=job.internal_id,
            state=job.state,
            video_url=job.video_url,
            error=job.error,
            provider=job.provider,
            model=job.model,
        )

    def _ensure_started(self) -> None:
        """Guard: raise if service not started."""
        if not self._started or self._cache is None:
            raise MediaError("Media service not started")

    @staticmethod
    def _parse_model_id(model_id: str) -> tuple[str, str]:
        """Parse 'provider:model_name' into (provider, model_name) for image providers.

        Raises:
            ProviderError: If the model ID format is invalid or provider unknown.
        """
        if ":" not in model_id:
            raise ProviderError(
                f"Invalid model ID '{model_id}' — expected format 'provider:model_name'"
            )
        provider, model_name = model_id.split(":", 1)
        if provider not in _PROVIDER_GENERATORS:
            raise ProviderError(
                f"Unknown image provider '{provider}'. Available: {', '.join(_PROVIDER_GENERATORS)}"
            )
        return provider, model_name

    @staticmethod
    def _parse_video_model_id(model_id: str) -> tuple[str, str]:
        """Parse 'provider:model_name' into (provider, model_name) for video providers.

        Raises:
            ProviderError: If the model ID format is invalid or provider unknown.
        """
        if ":" not in model_id:
            raise ProviderError(
                f"Invalid model ID '{model_id}' — expected format 'provider:model_name'"
            )
        provider, model_name = model_id.split(":", 1)
        if provider not in _VIDEO_PROVIDERS:
            raise ProviderError(
                f"Unknown video provider '{provider}'. Available: {', '.join(_VIDEO_PROVIDERS)}"
            )
        return provider, model_name

    @staticmethod
    async def _submit_video(
        provider: str,
        model_name: str,
        prompt: str,
        duration: int,
    ) -> str:
        """Dispatch video submission to the appropriate provider. Returns provider job ID."""
        if provider == "runway":
            return await submit_runway(prompt=prompt, model_name=model_name, duration=duration)
        if provider == "luma":
            return await submit_luma(prompt=prompt, model_name=model_name, duration=duration)
        raise ProviderError(f"No video handler for provider '{provider}'")

    @staticmethod
    def _get_video_poll_fn(provider: str):
        """Return the poll function for the given video provider."""
        if provider == "runway":
            return poll_runway
        if provider == "luma":
            return poll_luma
        raise ProviderError(f"No video poll handler for provider '{provider}'")

    @staticmethod
    async def _call_provider(
        provider: str,
        model_name: str,
        prompt: str,
        size: str,
        quality: str,
    ) -> GeneratedImage:
        """Dispatch to the appropriate provider function."""
        if provider == "openai":
            return await generate_openai(
                prompt=prompt,
                model_name=model_name,
                size=size,
                quality=quality,
            )
        if provider == "google":
            return await generate_google(
                prompt=prompt,
                model_name=model_name,
                size=size,
            )
        raise ProviderError(f"No handler for provider '{provider}'")
