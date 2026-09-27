"""The voice transport interface, and GPT-Live's documented HTTP/WebSocket protocol.

Everything that differs between voice providers lives in a transport; the
service asks it and never checks which provider is configured. The installed
OpenAI SDK has no Live surface. Keep this small transport private until an
upstream public adapter covers Live client delegation. No automatic creation
retries: the provider may already have allocated a billable session.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from loguru import logger

from assistant_runtime.app.voice.config import VoiceConfig
from assistant_runtime.app.voice.exceptions import VoiceError

# Known request rejections; timeout, proxy/nonstandard and other statuses stay unknown.
_REJECTED_CREATE_STATUSES = {400, 401, 402, 403, 404, 405, 409, 413, 415, 422, 429}


class VoiceTransport(Protocol):
    """What the voice service needs from a provider."""

    model: str  # reported on every call
    accepts_facts: bool  # host facts reach the call through the runtime
    reports_usage: bool  # usage() reports the provider's usage windows

    def status(self) -> dict:
        """Provider fields for the health check."""

    def credentials(self) -> str | None:
        """The key create, attach and abandon take; raises when no call can be created."""

    def session(
        self, instructions: str, voice: str, history: list[dict], *, conversation: bool
    ) -> dict:
        """The provider session for ``create``; history is role/text items, oldest first."""

    async def start(self) -> None:
        """Background start-up work, when voice is enabled."""

    async def usage(self) -> dict:
        """Only when ``reports_usage``."""

    async def create(self, api_key: str | None, session: dict, sdp: str) -> tuple[str, str]: ...

    async def attach(self, api_key: str | None, provider_id: str) -> Any: ...

    def abandon(self, provider_id: str, api_key: str | None) -> None:
        """Stop a session that was created but never attached; best effort, not awaited."""

    async def stop(self) -> None: ...


def build_transport(config: VoiceConfig) -> VoiceTransport:
    """The transport for the configured provider."""
    if config.provider == "codex":
        from assistant_runtime.app.voice._codex import CodexTransport

        return CodexTransport(
            config.codex_command,
            config.connect_timeout_seconds,
            usage_ceiling_percent=config.codex_usage_ceiling_percent,
            usage_check_seconds=config.codex_usage_check_seconds,
            close_timeout=config.close_timeout_seconds,
        )
    return LiveTransport(
        config.connect_timeout_seconds, model=config.model, api_key_env=config.api_key_env
    )


class LiveTransport:
    accepts_facts = False  # facts go through the browser's data channel
    reports_usage = False

    def __init__(
        self, timeout: float, *, model: str = "gpt-live-1", api_key_env: str = "OPENAI_API_KEY"
    ):
        self.timeout = timeout
        self.model = model
        self.api_key_env = api_key_env
        self.http = httpx.AsyncClient(timeout=timeout)
        self._background: set[asyncio.Task] = set()

    def status(self) -> dict:
        return {"configured": bool(os.getenv(self.api_key_env)), "model": self.model}

    def credentials(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise VoiceError(
                f"GPT-Live requires {self.api_key_env}; for the ChatGPT/Codex login "
                "set VOICE__PROVIDER=codex",
                503,
                allocation_status="rejected",
            )
        try:
            from websockets.asyncio.client import connect  # noqa: F401
        except ImportError as exc:
            raise VoiceError(
                "Install assistant-runtime[voice] to enable voice",
                503,
                allocation_status="rejected",
            ) from exc
        return key

    def session(
        self, instructions: str, voice: str, history: list[dict], *, conversation: bool
    ) -> dict:
        return {
            "model": self.model,
            "instructions": instructions,
            "audio": {"output": {"voice": voice}},
            # Live has no disabled delegation type. Runtime policy blocks dispatch.
            "delegation": {"type": "client"},
            **(
                {
                    "client": {
                        "data_channel": {
                            "allowed_client_events": [
                                "session.instructions.append",
                                "session.thinking.append",
                                "session.input_audio.mute",
                                "session.input_audio.unmute",
                            ]
                        }
                    }
                }
                if conversation
                else {}
            ),
            "input": [
                {
                    "type": "message",
                    "role": item["role"],
                    "content": [
                        {
                            "type": "input_text" if item["role"] == "user" else "output_text",
                            "text": item["text"],
                        }
                    ],
                }
                for item in history
            ],
            "store": False,
        }

    async def start(self) -> None:
        return None

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

    def abandon(self, session_id: str, api_key: str) -> None:
        # In the background, so a cancelled request cannot skip the hangup.
        task = asyncio.create_task(self._hangup(session_id, api_key))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _hangup(self, session_id: str, api_key: str) -> None:
        try:
            response = await self.http.post(
                f"https://api.openai.com/v1/live/sessions/{quote(session_id, safe='')}/hangup",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
        except Exception as exc:
            code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            logger.warning(
                "GPT-Live hangup of an unattached session failed: {} provider_status_code={}",
                type(exc).__name__,
                code,
            )

    async def stop(self) -> None:
        await asyncio.gather(*self._background, return_exceptions=True)
        await self.http.aclose()


async def send(connection: Any, event: dict) -> None:
    async with asyncio.timeout(10):
        await connection.send(json.dumps(event))
