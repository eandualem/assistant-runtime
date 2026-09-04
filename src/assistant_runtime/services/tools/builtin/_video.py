"""Video generation tools — video generation via external providers."""

from __future__ import annotations

from loguru import logger

from assistant_runtime.services.media.exceptions import ContentPolicyError, MediaError
from assistant_runtime.services.media.interface import MediaService
from assistant_runtime.services.tools._registry import ToolRegistry
from assistant_runtime.services.tools.models import ToolCategory, ToolDefinition


def register_video_tools(registry: ToolRegistry, media_service: MediaService) -> None:
    """Register video generation tools with closures capturing the media service."""

    async def generate_video(
        prompt: str,
        model: str = "",
        duration: int = 5,
    ) -> dict:
        """Generate a video from a text description."""
        try:
            result = await media_service.generate_video(
                prompt=prompt,
                model=model or None,
                duration=duration,
            )
            return {
                "success": True,
                "job_id": result.job_id,
                "status": result.status,
                "status_url": result.status_url,
                "provider": result.provider,
                "model": result.model,
                "message": "Video generation started. It typically takes 1-3 minutes.",
            }
        except ContentPolicyError as e:
            logger.warning("Video generation blocked by content policy", error=str(e))
            return {"success": False, "error": f"Content policy violation: {e}"}
        except MediaError as e:
            logger.error("Video generation failed", error=str(e))
            return {"success": False, "error": str(e)}

    registry.register_backend_tool(
        ToolDefinition(
            name="generate_video",
            description=(
                "Generate a video from a text description. This starts an async "
                "generation job that typically takes 1-3 minutes. Returns a job_id "
                "and status_url that can be polled for progress. Supports Runway "
                "and Luma AI providers."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Text description of the video to generate",
                    },
                    "model": {
                        "type": "string",
                        "description": (
                            "Model to use (e.g. 'runway:gen4-turbo', 'luma:ray-2'). "
                            "Defaults to the configured default video model."
                        ),
                        "default": "",
                    },
                    "duration": {
                        "type": "integer",
                        "description": (
                            "Video duration in seconds (typically 5 or 10). Defaults to 5."
                        ),
                        "default": 5,
                    },
                },
                "required": ["prompt"],
            },
            category=ToolCategory.BACKEND,
        ),
        generate_video,
    )

    logger.info("Registered video tools", count=1)
