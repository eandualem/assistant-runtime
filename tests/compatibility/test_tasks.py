"""A task investigates in the background while the conversation keeps being served."""

import asyncio

from pydantic_ai.messages import UserPromptPart

from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.base.events import EventHub

from .test_execution import assert_terminal, request


async def test_a_long_task_runs_while_another_request_is_served(runtime, script):
    investigation_may_finish = asyncio.Event()

    async def stream(messages, info):
        prompt = next(
            part.content
            for message in reversed(messages)
            for part in getattr(message, "parts", [])
            if isinstance(part, UserPromptPart)
        )
        if "investigate" in str(prompt):
            await investigation_may_finish.wait()
            yield "The parser leaks on empty input."
        else:
            yield "Hello."

    script.stream = stream
    hub, finished = EventHub(), []
    hub.subscribe(finished.append)
    tasks = TaskService(TasksConfig(enabled=True), runtime.streaming, events=hub)
    await tasks.start()
    try:
        record = await tasks.start_task("Please investigate the parser", parent_session_id="compat")
        await asyncio.sleep(0.05)
        assert (await tasks.get(record.id)).status == "running"

        # The conversation is served while the investigation is still running.
        assert_terminal([e async for e in runtime.streaming.stream_message(request())])
        assert (await tasks.get(record.id)).status == "running"

        investigation_may_finish.set()
        for _ in range(100):
            if finished:
                break
            await asyncio.sleep(0.01)
        assert finished[0]["status"] == "done"
        assert finished[0]["result"] == "The parser leaks on empty input."
        task_path = await runtime.sessions.get_message_path(record.session_id)
        assert [m["role"] for m in task_path] == ["user", "assistant"]
    finally:
        await tasks.stop()
