"""A task investigates in the background while the conversation keeps being served."""

import asyncio

from pydantic_ai.messages import UserPromptPart

from assistant_runtime.app.assistant.config import TunableOverrides
from assistant_runtime.app.tasks.config import TasksConfig
from assistant_runtime.app.tasks.interface import TaskService
from assistant_runtime.base.events import EventHub
from assistant_runtime.services.llm.config import LLMConfig

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


async def test_messages_to_a_persistent_agent_continue_one_conversation(runtime, script):
    seen: list[list[str]] = []

    async def stream(messages, info):
        prompts = [
            str(part.content)
            for message in messages
            for part in getattr(message, "parts", [])
            if isinstance(part, UserPromptPart)
        ]
        seen.append(prompts)
        yield f"Answer {len(prompts)}."

    script.stream = stream
    hub, finished = EventHub(), []
    hub.subscribe(finished.append)
    tasks = TaskService(TasksConfig(enabled=True), runtime.streaming, events=hub)
    await tasks.start()
    try:
        agent = await tasks.start_agent(subject="agent-a")
        await tasks.message_agent("First question", agent_id=agent.id, parent_session_id="main")
        await tasks.message_agent("Second question", agent_id=agent.id, parent_session_id="main")
        for _ in range(200):
            if len(finished) == 2:
                break
            await asyncio.sleep(0.01)
        assert [(e["status"], e["result"]) for e in finished] == [
            ("done", "Answer 1."),
            ("done", "Answer 2."),
        ]
        assert {e["session_id"] for e in finished} == {agent.session_id}
        # The second message is a turn in the same conversation: it sees the first.
        assert seen[1] == ["First question", "Second question"]
        path = await runtime.sessions.get_message_path(agent.session_id)
        assert [m["role"] for m in path] == ["user", "assistant", "user", "assistant"]
    finally:
        await tasks.stop()


async def test_an_agents_conversation_survives_session_cache_eviction(runtime, script):
    """Without a database the session cache is the conversation; an agent's stays cached."""
    seen: list[list[str]] = []

    async def stream(messages, info):
        seen.append(
            [
                str(part.content)
                for message in messages
                for part in getattr(message, "parts", [])
                if isinstance(part, UserPromptPart)
            ]
        )
        yield "Noted."

    script.stream = stream
    hub, finished = EventHub(), []
    hub.subscribe(finished.append)
    tasks = TaskService(TasksConfig(enabled=True), runtime.streaming, events=hub)
    await tasks.start()
    try:
        agent = await tasks.start_agent(subject="agent-a")
        for number, text in enumerate(("First question", "Second question"), start=1):
            if number == 2:
                for other in range(300):  # enough other sessions to force eviction
                    runtime.sessions.get_context(f"other-{other}")
            await tasks.message_agent(text, agent_id=agent.id)
            for _ in range(200):
                if len(finished) == number:
                    break
                await asyncio.sleep(0.01)
        assert seen[1] == ["First question", "Second question"]
    finally:
        await tasks.stop()


async def test_each_message_runs_with_the_agents_config_when_it_starts(
    runtime, script, monkeypatch
):
    """Clearing the config takes effect although the session remembers its last request config."""

    async def stream(messages, info):
        yield "Done."

    script.stream = stream
    models: list[str] = []
    run_message = runtime.streaming.run_message

    async def recording(request, **kwargs):
        result = await run_message(request, **kwargs)
        models.append(result.model)
        return result

    monkeypatch.setattr(runtime.streaming, "run_message", recording)
    hub, finished = EventHub(), []
    hub.subscribe(finished.append)
    tasks = TaskService(TasksConfig(enabled=True), runtime.streaming, events=hub)
    await tasks.start()
    try:
        agent = await tasks.start_agent(
            subject="agent-a", config=TunableOverrides(default_model="openai:gpt-5.1")
        )
        for number, text in enumerate(("First question", "Second question"), start=1):
            if number == 2:
                await tasks.configure_agent(
                    agent.id, TunableOverrides.model_validate({"default_model": None})
                )
            await tasks.message_agent(text, agent_id=agent.id)
            for _ in range(200):
                if len(finished) == number:
                    break
                await asyncio.sleep(0.01)
        assert [e["status"] for e in finished] == ["done", "done"]
        assert models == ["openai:gpt-5.1", LLMConfig().primary_model]
    finally:
        await tasks.stop()
