"""/host-state routes against a real HostStateService on an in-memory store."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from assistant_runtime.app.routes.host_state import router
from assistant_runtime.services.database.config import DatabaseConfig
from assistant_runtime.services.database.interface import DatabaseService
from assistant_runtime.services.host_state.config import HostStateConfig
from assistant_runtime.services.host_state.interface import HostStateService


async def test_put_get_list_and_delete_with_versions():
    service = HostStateService(HostStateConfig())
    await service.start()
    app = FastAPI()
    app.include_router(router)
    app.state.host_state_service = service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        created = await c.put(
            "/host-state/cursors/hub", json={"value": {"after": 3}, "expected_version": 0}
        )
        stale = await c.put(
            "/host-state/cursors/hub", json={"value": {"after": 2}, "expected_version": 0}
        )
        listed = (await c.get("/host-state/cursors")).json()
        wrong = await c.delete("/host-state/cursors/hub", params={"expected_version": 5})
        deleted = await c.delete("/host-state/cursors/hub", params={"expected_version": 1})
        gone = await c.get("/host-state/cursors/hub")
        bad = await c.get("/host-state/Bad%20Name")
    assert (created.status_code, created.json()["version"]) == (200, 1)
    assert created.json()["updated_by"] == "local"
    assert stale.status_code == 409
    assert ([e["value"] for e in listed["entries"]], listed["next_after"]) == ([{"after": 3}], None)
    assert (wrong.status_code, deleted.status_code, gone.status_code) == (409, 204, 404)
    assert bad.status_code == 422


async def test_a_database_lost_after_startup_is_a_503():
    from assistant_runtime.main import create_app

    database = DatabaseService(DatabaseConfig())
    session = AsyncMock()
    session.connection.side_effect = ConnectionRefusedError("Connection refused")
    database._session_factory = MagicMock()
    database._session_factory.return_value.__aenter__ = AsyncMock(return_value=session)
    database._session_factory.return_value.__aexit__ = AsyncMock(return_value=False)
    database._started = database._healthy = True
    service = HostStateService(HostStateConfig(), database)
    await service.start()
    app = create_app()
    app.state.host_state_service = service
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/api/host-state/cursors/hub")
    assert response.status_code == 503
    assert response.json() == {"error": "Database not reachable", "type": "DatabaseError"}
