"""Two independently configured hosts share execution without sharing prompt artifacts."""

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import create_app
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.request_context import get_current_profile_name


async def test_concurrent_profiles_scope_prompt_and_tool_edits(
    tmp_path, isolated_services, monkeypatch
):
    files = []
    for name in ("editor", "viewer"):
        path = tmp_path / f"{name}.toml"
        path.write_text(
            f'name = "{name}"\n[[artifacts]]\nname = "instructions"\n'
            f'default = "Persona {name}"\nrequired = true\n'
            '[[artifacts]]\nname = "notes"\n'
            '[artifacts.policy]\nassistant_edit = "autonomous"\nassistant_activate = true\n'
        )
        files.append(str(path))
    settings = isolated_services.model_copy(
        update={
            "assistant": isolated_services.assistant.model_copy(update={"profiles": files}),
            "tools": ToolConfig(
                builtin_tools=frozenset({"artifacts"}), provider_capabilities=frozenset()
            ),
        }
    )
    started = set()
    both = asyncio.Event()

    async def respond(messages, info):
        name = get_current_profile_name()
        assert name in ("editor", "viewer")
        prompt = info.instructions or ""
        assert f"Persona {name}" in prompt
        assert f"Persona {'viewer' if name == 'editor' else 'editor'}" not in prompt
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            started.add(name)
            if len(started) == 2:
                both.set()
            await both.wait()
            yield {
                0: DeltaToolCall(
                    "manage_artifacts",
                    json.dumps({"action": "update", "name": "notes", "content": f"Notes {name}"}),
                    tool_call_id=f"edit-{name}",
                )
            }
        else:
            assert returns[-1].content["success"]
            yield f"Done {name}"

    monkeypatch.setattr(
        LlmService,
        "_resolve_agent_model",
        lambda self, name: FunctionModel(stream_function=respond),
    )
    app = create_app(settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        async with asyncio.timeout(5):
            responses = await asyncio.gather(
                *(
                    client.post(
                        "/api/chat",
                        json=AssistantRequest(
                            id=f"message-{name}",
                            session_id=f"session-{name}",
                            profile=name,
                            content="Save a note",
                        ).model_dump(),
                    )
                    for name in ("editor", "viewer")
                )
            )
        assert [r.status_code for r in responses] == [200, 200], [r.text for r in responses]
        assert [r.json()["content"] for r in responses] == ["Done editor", "Done viewer"]
        for name in ("editor", "viewer"):
            scoped = app.state.artifact_service.for_profile(name)
            assert (await scoped.active_texts())["notes"] == f"Notes {name}"
        assert "notes" not in await app.state.artifact_service.active_texts()
        rejected = await client.post(
            "/api/chat",
            json={
                "id": "unknown",
                "session_id": "new-session",
                "content": "Hello",
                "profile": "missing",
            },
        )
        assert rejected.status_code == 409
        assert "Unknown assistant profile" in rejected.text
        assert (
            await app.state.assistant_service.get_session_store().get_context_if_exists_async(
                "new-session"
            )
            is None
        )


@pytest.mark.parametrize("profile", ["../private.toml", "/tmp/profile", "", "Uppercase", "a" * 65])
def test_request_cannot_load_arbitrary_profile_path(profile):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AssistantRequest(id="m", session_id="s", content="Hello", profile=profile)


@pytest.mark.parametrize("output_mode", ["text", "host_tools"])
async def test_profile_reaches_host_decisions_and_continuations(
    isolated_services, tmp_path, monkeypatch, output_mode
):
    path = tmp_path / "profile.toml"
    path.write_text('name="editor"\n[[artifacts]]\nname="instructions"\ndefault="Profile marker"\n')
    settings = isolated_services.model_copy(
        update={
            "assistant": isolated_services.assistant.model_copy(update={"profiles": [str(path)]})
        }
    )
    model_calls = []

    async def respond(messages, info):
        model_calls.append(get_current_profile_name())
        assert get_current_profile_name() == "editor"
        assert "Profile marker" in info.instructions
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if returns:
            yield "Applied"
        else:
            yield {0: DeltaToolCall("move", "{}", tool_call_id="move-1")}

    monkeypatch.setattr(
        LlmService,
        "_resolve_agent_model",
        lambda self, name: FunctionModel(stream_function=respond),
    )
    app = create_app(settings=settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post(
            "/api/chat",
            json={
                "id": "m",
                "session_id": "s",
                "content": "Move",
                "profile": "editor",
                "output_mode": output_mode,
                "host_context": {
                    "actions": [
                        {
                            "name": "move",
                            "description": "Move",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ]
                },
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["pending_tool_call"]["call_id"] == "move-1"
        receipt = {
            "id": "receipt",
            "session_id": "s",
            "content": "",
            "profile": "missing",
            "tool_call_id": "move-1",
            "tool_result": {"success": True},
        }
        rejected = await client.post("/api/chat", json=receipt)
        assert rejected.status_code == 409
        receipt["profile"] = "editor"
        completed = await client.post("/api/chat", json=receipt)
        assert completed.status_code == 200, completed.text
        assert completed.json()["content"] == ("Applied" if output_mode == "text" else None)
        assert model_calls == ["editor"] * (2 if output_mode == "text" else 1)
