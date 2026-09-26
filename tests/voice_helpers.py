"""Offline provider transport shared by unit and real-agent compatibility tests."""

import asyncio
import json
from unittest.mock import AsyncMock


class Connection:
    def __init__(self):
        self.frames = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.finalize = True

    def feed(self, kind, **data):
        self.frames.put_nowait(json.dumps({"type": kind, **data}))

    def __aiter__(self):
        return self

    async def __anext__(self):
        frame = await self.frames.get()
        if frame is None:
            raise StopAsyncIteration
        return frame

    async def send(self, frame):
        event = json.loads(frame)
        self.sent.append(event)
        if event["type"] == "session.close" and self.finalize:
            self.feed("session.closed", usage={"seconds": 12}, reason="close_requested")

    async def close(self):
        self.closed = True


class Transport:
    def __init__(self):
        self.connection = Connection()
        self.created = []
        self.attach = AsyncMock(return_value=self.connection)
        self.stopped = False

    async def create(self, key, config, sdp):
        self.created.append((key, config, sdp))
        return "live_provider", "answer"

    async def stop(self):
        self.stopped = True


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def delegate(connection, ident="item_1", text="Check availability", request=None):
    if text:
        connection.feed("session.input_transcript.delta", delta=text, start_ms=1, end_ms=2)
    delegation = {"id": ident, "target": "client"}
    if request is not None:  # the delegated request text some providers name
        delegation["input"] = request
    connection.feed("session.delegation.created", delegation=delegation)
