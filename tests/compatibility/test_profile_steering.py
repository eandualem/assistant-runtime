"""Explicit steering selectors survive delayed delivery without binding sessions."""

import asyncio

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.app.assistant.exceptions import SessionError
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import create_runtime
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.request_context import get_current_profile_name


@pytest.fixture
def profiles_settings(isolated_services, tmp_path):
    paths = []
    for name in ("editor", "viewer"):
        path = tmp_path / f"{name}.toml"
        path.write_text(
            f'name="{name}"\n[[artifacts]]\nname="instructions"\ndefault="Persona {name}"\n'
        )
        paths.append(str(path))
    return isolated_services.model_copy(
        update={
            "assistant": isolated_services.assistant.model_copy(
                update={"profile": paths[0], "profiles": paths[1:]}
            )
        }
    )


def request(ident, *, profile=None, steering=False, **fields):
    return AssistantRequest(
        id=ident,
        session_id="shared",
        content=ident,
        profile=profile,
        message_type="steering" if steering else "standard",
        **fields,
    )


def steering_text(messages):
    return "\n".join(
        part.content
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
        and isinstance(part.content, str)
        and "Additional user steering" in part.content
    )


async def test_cancelled_profile_steering_waits_through_other_profile_turn(
    profiles_settings, monkeypatch
):
    entered = asyncio.Event()
    observed = []

    async def respond(messages, info):
        observed.append((get_current_profile_name(), steering_text(messages)))
        if len(observed) == 1:
            entered.set()
            await asyncio.Event().wait()
        yield "Handled"

    monkeypatch.setattr(
        LlmService,
        "_resolve_agent_model",
        lambda self, name: FunctionModel(stream_function=respond),
    )
    async with create_runtime(settings=profiles_settings) as runtime:

        async def collect():
            return [
                event
                async for event in runtime.stream_message(request("initial", profile="editor"))
            ]

        task = asyncio.create_task(collect())
        try:
            async with asyncio.timeout(5):
                await entered.wait()
                assert (
                    await runtime.accept_steering(
                        request("editor-only", profile="editor", steering=True),
                        has_live_stream=True,
                    )
                    == "queued"
                )
                # Ingress uses SessionStore directly when draining its inbox.
                sessions = runtime._assistant_service.get_session_store()
                await sessions.queue_steering("shared", request("ingress", steering=True))
                assert await runtime.cancel_session("shared")
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert len(await sessions.list_pending_steering("shared")) == 2
        async with asyncio.timeout(5):
            await runtime.run_message(request("other", profile="viewer"))
        assert len(observed) == 2  # No extra paid follow-up for mismatched records.
        assert "editor-only" not in observed[1][1]
        assert "ingress" in observed[1][1]
        assert [row["id"] for row in await sessions.list_pending_steering("shared")] == [
            "editor-only"
        ]
        # Omitted turn selector resolves to the configured editor default.
        await runtime.run_message(request("retry"))
        assert len(observed) == 3
        assert "editor-only" in observed[2][1]
        assert not await sessions.list_pending_steering("shared")


async def test_pending_host_continuation_consumes_only_matching_steering(
    profiles_settings, monkeypatch
):
    observed = []

    async def respond(messages, info):
        observed.append((get_current_profile_name(), steering_text(messages)))
        if len(observed) == 1:
            yield {0: DeltaToolCall("select_item", "{}", tool_call_id="select-1")}
        else:
            yield "Handled"

    monkeypatch.setattr(
        LlmService,
        "_resolve_agent_model",
        lambda self, name: FunctionModel(stream_function=respond),
    )
    async with create_runtime(settings=profiles_settings) as runtime:
        initial = await runtime.run_message(
            request(
                "initial",
                profile="editor",
                host_context={
                    "actions": [
                        {
                            "name": "select_item",
                            "description": "Select an item",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                },
            )
        )
        assert initial.pending_tool_call["call_id"] == "select-1"
        for profile in ("editor", "viewer"):
            assert (
                await runtime.accept_steering(
                    request(f"{profile}-only", profile=profile, steering=True),
                    has_live_stream=False,
                )
                == "queued"
            )
        async with asyncio.timeout(5):
            await runtime.run_message(
                request(
                    "receipt", profile="editor", tool_call_id="select-1", tool_result={"ok": True}
                )
            )
        assert len(observed) == 2
        assert "editor-only" in observed[1][1]
        assert "viewer-only" not in observed[1][1]
        await runtime.run_message(request("next", profile="viewer"))
        assert len(observed) == 3
        assert "viewer-only" in observed[2][1]


async def test_promoted_steering_uses_its_profile_and_rejects_selector_rebinding(
    profiles_settings, monkeypatch
):
    observed = []

    async def respond(messages, info):
        observed.append((info.instructions, steering_text(messages)))
        yield "Handled"

    monkeypatch.setattr(
        LlmService,
        "_resolve_agent_model",
        lambda self, name: FunctionModel(stream_function=respond),
    )
    async with create_runtime(settings=profiles_settings) as runtime:
        await runtime.run_message(request("initial"))
        steering = request("viewer-only", profile="viewer", steering=True)
        assert await runtime.accept_steering(steering, has_live_stream=False) == "promoted"
        with pytest.raises(SessionError, match="original profile selector"):
            await runtime.run_message(steering.model_copy(update={"profile": "editor"}))
        assert len(observed) == 1
        await runtime.run_message(steering)
        assert len(observed) == 2
        assert "Persona viewer" in observed[1][0]
        assert "viewer-only" in observed[1][1]
