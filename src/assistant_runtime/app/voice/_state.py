"""Process-local connections and serializable call state."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from assistant_runtime.app.voice.models import VoiceOffer
from assistant_runtime.principal import Principal


@dataclass
class VoiceCall:
    id: str
    offer: VoiceOffer
    principal: Principal
    lease: str
    model: str
    voice: str
    provider_id: str = ""
    status: str = "connecting"
    reason: str | None = None
    created_at: float = field(default_factory=time.time)
    finalized: bool = False
    usage: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    transcript_chars: int = 0
    seen_events: set[str] = field(default_factory=set)
    delegations: dict[str, dict] = field(default_factory=dict)
    active_delegation: str | None = None
    pending: dict | None = None
    cursor: int = 0
    events: deque = field(default_factory=deque)
    connection: Any = None
    task: asyncio.Task | None = None
    work: asyncio.Task | None = None
    work_continuation: bool = False
    cancelling: bool = False
    cancel_task: asyncio.Task | None = None
    deferred_delegation: str | None = None
    control_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=32))
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    dirty: asyncio.Event = field(default_factory=asyncio.Event)
    stop_requested: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self) -> dict:
        return {
            "call_id": self.id,
            "session_id": self.offer.session_id,
            "owner_id": self.principal.id,
            "provider_session_id": self.provider_id,
            "model": self.model,
            "voice": self.voice,
            "status": self.status,
            "reason": self.reason,
            "created_at": self.created_at,
            "finalized": self.finalized,
            "usage": dict(self.usage),
            "transcript": list(self.transcript),
            "delegations": {k: dict(v) for k, v in self.delegations.items()},
            "active_delegation": self.active_delegation,
            "pending_tool_call": self.pending,
            "cursor": self.cursor,
        }
