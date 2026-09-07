"""Backbone telemetry and delivery status tools."""

from __future__ import annotations

from typing import Any

from assistant_runtime.services.tools.providers.backbone._client import (
    backbone_detail,
    backbone_error,
    backbone_request,
)

_MAX_LIMIT = 500


def _validate_limit(limit: int) -> str | None:
    """Validate a list endpoint limit."""
    if not 1 <= limit <= _MAX_LIMIT:
        return f"limit must be between 1 and {_MAX_LIMIT}"
    return None


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], int] | tuple[None, None]:
    """Extract a standard ListEnvelope payload."""
    if not isinstance(payload, dict):
        return (None, None)

    items = payload.get("items")
    total = payload.get("total")
    if not isinstance(items, list) or not isinstance(total, int):
        return (None, None)

    normalized_items = [item for item in items if isinstance(item, dict)]
    return (normalized_items, total)


async def get_delivery_status() -> dict[str, Any]:
    """Fetch aggregated delivery health from the backbone."""
    status, data = await backbone_request("GET", "/api/deliveries/stats")
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status != 200 or not isinstance(data, dict):
        detail = backbone_detail(data) if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}

    return {
        "total": int(data.get("total", 0)),
        "delivered": int(data.get("delivered", 0)),
        "failed": int(data.get("failed", 0)),
        "deferred": int(data.get("deferred", 0)),
        "offline": int(data.get("offline", 0)),
        "success": True,
    }


async def get_recent_deliveries(limit: int = 20) -> dict[str, Any]:
    """Fetch recent delivery attempts and outcomes."""
    error = _validate_limit(limit)
    if error:
        return {"error": error, "success": False}

    status, data = await backbone_request(
        "GET",
        "/api/deliveries",
        params={"limit": str(limit)},
    )
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status != 200:
        detail = backbone_detail(data) if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}

    items, total = _extract_items(data)
    if items is None or total is None:
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "deliveries": items,
        "count": len(items),
        "total": total,
        "success": True,
    }


async def get_failed_deliveries(limit: int = 20) -> dict[str, Any]:
    """Fetch failed, deferred, or offline deliveries."""
    error = _validate_limit(limit)
    if error:
        return {"error": error, "success": False}

    status, data = await backbone_request(
        "GET",
        "/api/deliveries/failed",
        params={"limit": str(limit)},
    )
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status != 200:
        detail = backbone_detail(data) if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}

    items, total = _extract_items(data)
    if items is None or total is None:
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "deliveries": items,
        "count": len(items),
        "total": total,
        "success": True,
    }


async def get_agent_activity(
    session_name: str,
    limit: int = 20,
    since: float | None = None,
) -> dict[str, Any]:
    """Fetch recent activity for a specific agent session."""
    if not session_name or not session_name.strip():
        return {"error": "session_name cannot be empty", "success": False}

    error = _validate_limit(limit)
    if error:
        return {"error": error, "success": False}

    params: dict[str, str] = {"limit": str(limit)}
    if since is not None:
        params["since"] = str(since)

    session = session_name.strip()
    status, data = await backbone_request(
        "GET",
        f"/api/agents/{session}/activity",
        params=params,
    )
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status == 404:
        detail = data.get("detail", "Not found") if isinstance(data, dict) else "Not found"
        return {"error": f"Agent activity not found: {detail}", "success": False}
    if status != 200:
        detail = backbone_detail(data) if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}

    items, total = _extract_items(data)
    if items is None or total is None:
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "session_name": session,
        "events": items,
        "count": len(items),
        "total": total,
        "success": True,
    }


async def get_activity_timeline(limit: int = 20, offset: int = 0) -> dict[str, Any]:
    """Fetch the merged system-wide activity timeline."""
    error = _validate_limit(limit)
    if error:
        return {"error": error, "success": False}
    if offset < 0:
        return {"error": "offset must be greater than or equal to 0", "success": False}

    status, data = await backbone_request(
        "GET",
        "/api/activity/timeline",
        params={"limit": str(limit), "offset": str(offset)},
    )
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status != 200:
        detail = backbone_detail(data) if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}

    items, total = _extract_items(data)
    if items is None or total is None:
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "events": items,
        "count": len(items),
        "total": total,
        "offset": offset,
        "success": True,
    }


class BackboneActivity:
    """The activity capability served by this provider (see ``capabilities.activity``)."""

    async def get_delivery_status(self) -> dict[str, Any]:
        return await get_delivery_status()

    async def get_recent_deliveries(self, limit: int = 20) -> dict[str, Any]:
        return await get_recent_deliveries(limit=limit)

    async def get_failed_deliveries(self, limit: int = 20) -> dict[str, Any]:
        return await get_failed_deliveries(limit=limit)

    async def get_agent_activity(
        self,
        session_name: str,
        limit: int = 20,
        since: float | None = None,
    ) -> dict[str, Any]:
        return await get_agent_activity(session_name=session_name, limit=limit, since=since)

    async def get_activity_timeline(self, limit: int = 20, offset: int = 0) -> dict[str, Any]:
        return await get_activity_timeline(limit=limit, offset=offset)
