"""Documented GPT-Live HTTP/WebSocket protocol (not the Realtime API).

The installed OpenAI SDK has no Live surface. Keep this small transport private
until an upstream public adapter covers Live client delegation. No automatic
creation retries: the provider may already have allocated a billable session.
"""

import asyncio
import json
import re
from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger

from assistant_runtime.app.voice.exceptions import VoiceError

# Known request rejections; timeout, proxy/nonstandard and other statuses stay unknown.
_REJECTED_CREATE_STATUSES = {400, 401, 402, 403, 404, 405, 409, 413, 415, 422, 429}


class LiveTransport:
    def __init__(self, timeout: float):
        self.timeout = timeout
        self.http = httpx.AsyncClient(timeout=timeout)

    async def create(self, api_key: str, session: dict, sdp: str) -> tuple[str, str]:
        try:
            response = await self.http.post(
                "https://api.openai.com/v1/live/sessions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"session": session, "transport": {"type": "webrtc", "sdp": sdp}},
            )
            response.raise_for_status()
            data = response.json()
            session_id, answer = data["session"]["id"], data["transport"]["sdp"]
            if (
                not isinstance(session_id, str)
                or not session_id
                or not isinstance(answer, str)
                or not answer
            ):
                raise ValueError("Invalid session response")
            return session_id, answer
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            message = (
                "GPT-Live rejected the API credentials or model access"
                if code in (401, 403)
                else "GPT-Live session creation failed"
            )
            allocation = "rejected" if code in _REJECTED_CREATE_STATUSES else "unknown"
            request_id = exc.response.headers.get("x-request-id", "")
            # Accept only the provider's opaque request-id form, never arbitrary headers.
            if not re.fullmatch(r"req_[0-9a-fA-F]{32}", request_id):
                request_id = None
            logger.warning(
                "GPT-Live create HTTP failure: provider_status_code={} allocation_status={} "
                "provider_request_id={}",
                code,
                allocation,
                request_id,
            )
            raise VoiceError(
                message,
                502,
                allocation_status=allocation,
                provider_status_code=code,
                provider_request_id=request_id,
            ) from exc
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("GPT-Live create failed: allocation_status=unknown")
            raise VoiceError(
                "GPT-Live session creation failed; allocation may be unconfirmed",
                502,
                allocation_status="unknown",
            ) from exc

    async def attach(self, api_key: str, session_id: str) -> Any:
        # Import only when voice is enabled, with a declared optional extra.
        from websockets.asyncio.client import connect

        return await connect(
            f"wss://api.openai.com/v1/live/sessions/{quote(session_id, safe='')}/attach",
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=self.timeout,
            close_timeout=2,
            max_size=2**20,
            max_queue=16,
        )

    async def stop(self) -> None:
        await self.http.aclose()


async def send(connection: Any, event: dict) -> None:
    async with asyncio.timeout(10):
        await connection.send(json.dumps(event))
