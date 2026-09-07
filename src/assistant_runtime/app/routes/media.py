"""Media serving endpoints — serves cached images and video job status."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from assistant_runtime.services.media.deps import MediaServiceDep

router = APIRouter(prefix="/media", tags=["media"])


@router.get("/video/{job_id}")
async def get_video_status(job_id: str, media_service: MediaServiceDep) -> dict:
    """Get the current status of a video generation job."""
    status = media_service.get_video_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Video job not found")
    return status.model_dump()


@router.get("/{image_id}")
async def get_image(image_id: str, media_service: MediaServiceDep) -> Response:
    """Serve a cached generated image by ID."""
    result = media_service.get_cached_image(image_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Image not found or expired")
    image_bytes, mime_type = result
    return Response(
        content=image_bytes,
        media_type=mime_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )
