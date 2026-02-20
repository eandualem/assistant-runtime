"""Media generation tools — image generation via external providers."""

from __future__ import annotations

from loguru import logger

from lovely_assistant.services.media.exceptions import ContentPolicyError, MediaError
from lovely_assistant.services.media.interface import MediaService
from lovely_assistant.services.tools._registry import ToolRegistry
from lovely_assistant.services.tools.models import ToolCategory, ToolDefinition


def register_media_tools(registry: ToolRegistry, media_service: MediaService) -> None:
    """Register media generation tools with closures capturing the media service."""

    async def generate_image(
        prompt: str,
        model: str = "",
        size: str = "",
        quality: str = "",
    ) -> dict:
        """Generate an image from a text description."""
        try:
            result = await media_service.generate_image(
                prompt=prompt,
                model=model or None,
                size=size or None,
                quality=quality or None,
            )
            return {
                "success": True,
                "image_url": result.url,
                "image_id": result.image_id,
                "model": result.model,
                "provider": result.provider,
            }
        except ContentPolicyError as e:
            logger.warning("Image generation blocked by content policy", error=str(e))
            return {"success": False, "error": f"Content policy violation: {e}"}
        except MediaError as e:
            logger.error("Image generation failed", error=str(e))
            return {"success": False, "error": str(e)}

    registry.register_backend_tool(
        ToolDefinition(
            name="generate_image",
            description=(
                "Generate an image from a text description. Returns a URL to the "
                "generated image that can be displayed in chat. Supports multiple "
                "providers and models. Use the image_url from the result to show "
                "the image to the user."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the image to generate",
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Model to use (e.g. 'openai:gpt-image-1', "
                            "'google:imagen-4.0-generate-preview-06-06'). "
                            "Defaults to the configured default model."
                        ),
                        "default": "",
                    },
                    "size": {
                        "type": "string",
                        "description": (
                            "Image size (e.g. '1024x1024', '1536x1024', '1024x1536'). "
                            "Defaults to 1024x1024."
                        ),
                        "default": "",
                    },
                    "quality": {
                        "type": "string",
                        "description": (
                            "Quality level (e.g. 'low', 'medium', 'high'). "
                            "Defaults to medium. Only applies to OpenAI models."
                        ),
                        "default": "",
                    },
                },
                "required": ["prompt"],
            },
            category=ToolCategory.BACKEND,
        ),
        generate_image,
    )

    logger.info("Registered media tools", count=1)
