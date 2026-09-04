"""Backbone swarm management and monitoring tools."""

from __future__ import annotations

from typing import Any

from assistant_runtime.services.tools.capabilities.workgroups import (
    _SWARM_PHASES,
    _SWARM_WORKER_ROLES,
    _SWARM_WORKER_STATUSES,
    _WORKER_REQUIRED_FIELDS,
)
from assistant_runtime.services.tools.providers.backbone._client import (
    backbone_error,
    backbone_request,
)


def _validate_required_text(value: str, field_name: str) -> str | None:
    """Validate a required string parameter."""
    if not value or not value.strip():
        return f"{field_name} cannot be empty"
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


def _validate_phase(status: str | None) -> tuple[str | None, str | None]:
    """Validate the user-facing status filter, which maps to swarm phase."""
    if status is None:
        return (None, None)
    if not status.strip():
        return (None, "status cannot be empty")

    normalized = status.strip()
    if normalized not in _SWARM_PHASES:
        allowed = ", ".join(sorted(_SWARM_PHASES))
        return (None, f"status must be one of: {allowed}")

    return (normalized, None)


def _validate_workers(
    workers: list[dict[str, str]] | None,
) -> tuple[list[dict[str, str]] | None, str | None]:
    """Validate worker payloads for swarm creation."""
    if workers is None:
        return (None, None)

    normalized_workers: list[dict[str, str]] = []
    for index, worker in enumerate(workers):
        if not isinstance(worker, dict):
            return (None, f"workers[{index}] must be an object")

        normalized_worker: dict[str, str] = {}
        for field_name in _WORKER_REQUIRED_FIELDS:
            value = worker.get(field_name)
            if not isinstance(value, str) or not value.strip():
                return (None, f"workers[{index}].{field_name} cannot be empty")
            normalized_worker[field_name] = value.strip()

        role = normalized_worker["role"]
        if role not in _SWARM_WORKER_ROLES:
            allowed = ", ".join(sorted(_SWARM_WORKER_ROLES))
            return (None, f"workers[{index}].role must be one of: {allowed}")

        normalized_workers.append(normalized_worker)

    return (normalized_workers, None)


async def create_swarm(
    repo: str,
    task_id: str | None = None,
    coding_agent_session: str = "",
    workers: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create a swarm in the backbone registry."""
    repo_error = _validate_required_text(repo, "repo")
    if repo_error:
        return {"error": repo_error, "success": False}

    session_error = _validate_required_text(coding_agent_session, "coding_agent_session")
    if session_error:
        return {"error": session_error, "success": False}

    normalized_workers, workers_error = _validate_workers(workers)
    if workers_error:
        return {"error": workers_error, "success": False}

    normalized_repo = repo.strip()
    normalized_session = coding_agent_session.strip()
    normalized_task_id = task_id.strip() if task_id and task_id.strip() else None

    body: dict[str, Any] = {
        "repo": normalized_repo,
        "coding_agent_session": normalized_session,
    }
    if normalized_task_id is not None:
        body["task_id"] = normalized_task_id
    if normalized_workers is not None:
        body["workers"] = normalized_workers

    status, data = await backbone_request("POST", "/api/swarms", json_body=body)
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status != 201:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}
    if not isinstance(data, dict) or not isinstance(data.get("swarm_id"), str):
        return {"error": "Unexpected backbone response format", "success": False}

    worker_list = normalized_workers or []
    return {
        "swarm_id": data["swarm_id"],
        "repo": normalized_repo,
        "task_id": normalized_task_id,
        "coding_agent_session": normalized_session,
        "workers": worker_list,
        "worker_count": len(worker_list),
        "success": True,
    }


async def list_swarms(repo: str | None = None, status: str | None = None) -> dict[str, Any]:
    """List swarms with optional repo and phase filters."""
    params: dict[str, str] = {}

    if repo is not None:
        if not repo.strip():
            return {"error": "repo cannot be empty", "success": False}
        params["repo"] = repo.strip()

    normalized_phase, phase_error = _validate_phase(status)
    if phase_error:
        return {"error": phase_error, "success": False}
    if normalized_phase is not None:
        params["phase"] = normalized_phase

    status_code, data = await backbone_request("GET", "/api/swarms", params=params or None)
    if status_code == -1:
        return {"error": backbone_error(data), "success": False}
    if status_code != 200:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status_code}): {detail}", "success": False}

    items, total = _extract_items(data)
    if items is None or total is None:
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "swarms": items,
        "count": len(items),
        "total": total,
        "success": True,
    }


async def get_swarm_detail(swarm_id: str) -> dict[str, Any]:
    """Get full detail for a single swarm."""
    error = _validate_required_text(swarm_id, "swarm_id")
    if error:
        return {"error": error, "success": False}

    normalized_swarm_id = swarm_id.strip()
    status, data = await backbone_request("GET", f"/api/swarms/{normalized_swarm_id}")
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status == 404:
        return {"error": f"Swarm '{normalized_swarm_id}' not found", "success": False}
    if status != 200:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}
    if not isinstance(data, dict):
        return {"error": "Unexpected backbone response format", "success": False}

    return {"swarm": data, "success": True}


async def update_worker_status(
    swarm_id: str,
    worker_name: str,
    status: str,
    pr_number: int | None = None,
) -> dict[str, Any]:
    """Update a swarm worker's status."""
    swarm_error = _validate_required_text(swarm_id, "swarm_id")
    if swarm_error:
        return {"error": swarm_error, "success": False}

    worker_error = _validate_required_text(worker_name, "worker_name")
    if worker_error:
        return {"error": worker_error, "success": False}

    status_error = _validate_required_text(status, "status")
    if status_error:
        return {"error": status_error, "success": False}

    normalized_status = status.strip()
    if normalized_status not in _SWARM_WORKER_STATUSES:
        allowed = ", ".join(sorted(_SWARM_WORKER_STATUSES))
        return {"error": f"status must be one of: {allowed}", "success": False}

    if pr_number is not None and pr_number < 1:
        return {"error": "pr_number must be greater than or equal to 1", "success": False}

    normalized_swarm_id = swarm_id.strip()
    normalized_worker_name = worker_name.strip()

    body: dict[str, Any] = {"status": normalized_status}
    if pr_number is not None:
        body["pr_number"] = pr_number

    response_status, data = await backbone_request(
        "POST",
        f"/api/swarms/{normalized_swarm_id}/workers/{normalized_worker_name}/status",
        json_body=body,
    )
    if response_status == -1:
        return {"error": backbone_error(data), "success": False}
    if response_status == 404:
        return {"error": f"Swarm worker '{normalized_worker_name}' not found", "success": False}
    if response_status != 200:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({response_status}): {detail}", "success": False}
    if not isinstance(data, dict):
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "swarm": data,
        "worker_name": normalized_worker_name,
        "status": normalized_status,
        "success": True,
    }


async def broadcast_to_swarm(swarm_id: str, from_entity: str, message: str) -> dict[str, Any]:
    """Broadcast a message to every worker in a swarm."""
    swarm_error = _validate_required_text(swarm_id, "swarm_id")
    if swarm_error:
        return {"error": swarm_error, "success": False}

    entity_error = _validate_required_text(from_entity, "from_entity")
    if entity_error:
        return {"error": entity_error, "success": False}

    message_error = _validate_required_text(message, "message")
    if message_error:
        return {"error": message_error, "success": False}

    normalized_swarm_id = swarm_id.strip()
    status, data = await backbone_request(
        "POST",
        f"/api/swarms/{normalized_swarm_id}/broadcast",
        json_body={"from_entity": from_entity.strip(), "message": message.strip()},
    )
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status == 404:
        return {"error": f"Swarm '{normalized_swarm_id}' not found", "success": False}
    if status != 200:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}
    if not isinstance(data, dict):
        return {"error": "Unexpected backbone response format", "success": False}

    return {
        "swarm_id": normalized_swarm_id,
        "ok": bool(data.get("ok", False)),
        "message_id": data.get("message_id"),
        "delivered": int(data.get("delivered", 0)),
        "failed": int(data.get("failed", 0)),
        "total": int(data.get("total", 0)),
        "success": True,
    }


async def complete_swarm(swarm_id: str) -> dict[str, Any]:
    """Mark a swarm as completed/cleaned up."""
    error = _validate_required_text(swarm_id, "swarm_id")
    if error:
        return {"error": error, "success": False}

    normalized_swarm_id = swarm_id.strip()
    status, data = await backbone_request("DELETE", f"/api/swarms/{normalized_swarm_id}")
    if status == -1:
        return {"error": backbone_error(data), "success": False}
    if status == 404:
        return {"error": f"Swarm '{normalized_swarm_id}' not found", "success": False}
    if status != 200:
        detail = data.get("detail", "Unknown error") if isinstance(data, dict) else str(data)
        return {"error": f"Backbone API error ({status}): {detail}", "success": False}
    if not isinstance(data, dict):
        return {"error": "Unexpected backbone response format", "success": False}

    return {"swarm": data, "success": True}


class BackboneWorkgroups:
    """The workgroups capability served by this provider (see ``capabilities.workgroups``)."""

    async def create_swarm(
        self,
        repo: str,
        task_id: str | None = None,
        coding_agent_session: str = "",
        workers: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        return await create_swarm(
            repo=repo, task_id=task_id, coding_agent_session=coding_agent_session, workers=workers
        )

    async def list_swarms(
        self, repo: str | None = None, status: str | None = None
    ) -> dict[str, Any]:
        return await list_swarms(repo=repo, status=status)

    async def get_swarm_detail(self, swarm_id: str) -> dict[str, Any]:
        return await get_swarm_detail(swarm_id=swarm_id)

    async def update_worker_status(
        self,
        swarm_id: str,
        worker_name: str,
        status: str,
        pr_number: int | None = None,
    ) -> dict[str, Any]:
        return await update_worker_status(
            swarm_id=swarm_id, worker_name=worker_name, status=status, pr_number=pr_number
        )

    async def broadcast_to_swarm(
        self, swarm_id: str, from_entity: str, message: str
    ) -> dict[str, Any]:
        return await broadcast_to_swarm(swarm_id=swarm_id, from_entity=from_entity, message=message)

    async def complete_swarm(self, swarm_id: str) -> dict[str, Any]:
        return await complete_swarm(swarm_id=swarm_id)
