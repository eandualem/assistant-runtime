"""Image generation provider implementations.

Each provider is a standalone async function — easy to add new providers
by adding another function and a routing entry in MediaService.
"""

from __future__ import annotations

import base64
import os

from loguru import logger

from lovely_assistant.services.media.exceptions import (
    ContentPolicyError,
    ProviderError,
    ProviderNotConfiguredError,
)
from lovely_assistant.services.media.models import GeneratedImage

# Size string → Google aspect ratio mapping
_SIZE_TO_ASPECT_RATIO: dict[str, str] = {
    "1024x1024": "1:1",
    "1536x1024": "3:2",
    "1024x1536": "2:3",
}


async def generate_openai(
    prompt: str,
    model_name: str,
    size: str,
    quality: str,
) -> GeneratedImage:
    """Generate an image using OpenAI's images API.

    Args:
        prompt: Text description of the image to generate.
        model_name: Model name without provider prefix (e.g. "gpt-image-1").
        size: Image dimensions (e.g. "1024x1024").
        quality: Quality level (e.g. "medium", "high").

    Returns:
        GeneratedImage with raw PNG bytes.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProviderNotConfiguredError("OPENAI_API_KEY environment variable is not set")

    try:
        from openai import AsyncOpenAI, BadRequestError, OpenAIError

        client = AsyncOpenAI(api_key=api_key)
        response = await client.images.generate(
            prompt=prompt,
            model=model_name,
            size=size,
            quality=quality,
            response_format="b64_json",
            n=1,
        )

        b64_data = response.data[0].b64_json
        if not b64_data:
            raise ProviderError("OpenAI returned empty image data")

        image_bytes = base64.b64decode(b64_data)
        logger.info(
            "OpenAI image generated",
            model=model_name,
            size=size,
            bytes=len(image_bytes),
        )
        return GeneratedImage(
            image_bytes=image_bytes,
            mime_type="image/png",
            provider="openai",
            model=f"openai:{model_name}",
        )

    except BadRequestError as e:
        error_msg = str(e)
        if "content_policy" in error_msg.lower() or "safety" in error_msg.lower():
            raise ContentPolicyError(f"OpenAI content policy rejection: {error_msg}") from e
        raise ProviderError(f"OpenAI API error: {error_msg}") from e
    except OpenAIError as e:
        raise ProviderError(f"OpenAI API error: {e}") from e


async def generate_google(
    prompt: str,
    model_name: str,
    size: str,
) -> GeneratedImage:
    """Generate an image using Google's Imagen API.

    Args:
        prompt: Text description of the image to generate.
        model_name: Model name without provider prefix (e.g. "imagen-4.0-generate-preview-06-06").
        size: Image dimensions (e.g. "1024x1024") — mapped to aspect ratio.

    Returns:
        GeneratedImage with raw image bytes.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ProviderNotConfiguredError("GOOGLE_API_KEY environment variable is not set")

    aspect_ratio = _SIZE_TO_ASPECT_RATIO.get(size, "1:1")

    try:
        from google import genai
        from google.genai.types import GenerateImagesConfig

        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
            ),
        )

        if not response.generated_images:
            raise ProviderError("Google returned no images")

        image_data = response.generated_images[0].image
        if not image_data or not image_data.image_bytes:
            raise ProviderError("Google returned empty image data")

        image_bytes = image_data.image_bytes
        # Google Imagen returns PNG by default
        mime_type = getattr(image_data, "mime_type", None) or "image/png"

        logger.info(
            "Google image generated",
            model=model_name,
            aspect_ratio=aspect_ratio,
            bytes=len(image_bytes),
        )
        return GeneratedImage(
            image_bytes=image_bytes,
            mime_type=mime_type,
            provider="google",
            model=f"google:{model_name}",
        )

    except ImportError as e:
        raise ProviderError(f"Google GenAI SDK not available: {e}") from e
    except Exception as e:
        error_msg = str(e)
        if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
            raise ContentPolicyError(f"Google content policy rejection: {error_msg}") from e
        # Don't re-wrap our own exceptions
        if isinstance(e, (ProviderError, ContentPolicyError, ProviderNotConfiguredError)):
            raise
        raise ProviderError(f"Google API error: {error_msg}") from e
