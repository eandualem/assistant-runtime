"""Opt-in provider/HTTP smoke; never collected by the default tests/ CI gate."""

import os
from time import perf_counter
from uuid import uuid4

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_SMOKE") != "1", reason="Set RUN_MODEL_SMOKE=1 for a live model call"
)


async def test_model_turn_is_saved(record_property):
    """A real provider completes a turn and the public API returns its saved row."""
    session_id = f"smoke-{uuid4().hex}"
    base_url = os.environ["SMOKE_RUNTIME_URL"].rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        started = perf_counter()
        completed = False
        try:
            response = await client.post(
                "/api/chat",
                json={
                    "id": f"smoke-{uuid4().hex}",
                    "session_id": session_id,
                    "content": "Reply with exactly SMOKE_OK. Do not use tools.",
                    "config": {"max_turns": 2, "enable_working_memory": False},
                },
            )
            response.raise_for_status()
            completed = True
            result = response.json()
            assert "SMOKE_OK" in result["content"]
            assert result["pending_tool_call"] is None
            assert result["session_id"] == session_id
            assert result["message_id"]

            saved = await client.get(f"/api/sessions/{session_id}/messages")
            saved.raise_for_status()
            assistant = next(row for row in saved.json() if row["id"] == result["message_id"])
            assert assistant["role"] == "assistant"
            assert assistant["content"] == result["content"]
            record_property("model", result["model"])
            record_property("elapsed_seconds", round(perf_counter() - started, 3))
            usage = assistant.get("usage")
            assert isinstance(usage, dict), "Saved assistant message has no usage metadata"
            record_property("usage", str(usage))
        finally:
            if completed:
                # A successful non-streaming response means turn cleanup completed.
                cleanup = await client.delete(f"/api/sessions/{session_id}")
                assert cleanup.status_code in {200, 404}, (
                    f"Smoke session cleanup failed: {session_id}"
                )
            else:
                # The HTTP cancellation route requests cancellation but cannot
                # confirm draining. Preserve state when the POST result is unknown.
                try:
                    cancel = await client.post(f"/api/chat/{session_id}/cancel")
                    cancel.raise_for_status()
                except httpx.HTTPError as exc:
                    raise AssertionError(
                        f"Turn/cancellation status unknown; inspect smoke session {session_id}"
                    ) from exc
                pytest.fail(
                    f"Turn did not complete successfully; cancellation requested. "
                    f"Cleanup is unconfirmed; inspect smoke session {session_id}"
                )
