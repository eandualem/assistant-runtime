"""A host module composes the runtime exclusively through public interfaces."""

from __future__ import annotations

import asyncio
import importlib.util
import sys

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError
from pydantic_ai.messages import SystemPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from assistant_runtime.app.assistant.exceptions import AgentRunError
from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.main import AssistantDefinition, create_app, create_asgi_app, create_runtime
from assistant_runtime.services.llm.interface import LlmService
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.exceptions import ToolError
from assistant_runtime.services.tools.interface import ToolService

HOST_MODULE = """\
import asyncio
from dataclasses import dataclass
from pydantic_ai import RunContext, Tool
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.toolsets import FunctionToolset
from assistant_runtime.main import AssistantDefinition

created_deps = []
seen_deps = []

@dataclass
class BusinessDeps:
    request_id: str

async def dependencies(request):
    await asyncio.sleep(0)
    deps = BusinessDeps(request.id)
    created_deps.append(deps)
    return deps

def business_tool(ctx: RunContext[BusinessDeps], item: str) -> str:
    seen_deps.append(ctx.deps)
    return f"{ctx.deps.request_id}:{item}"

toolset = FunctionToolset()

@toolset.tool
def currency(ctx: RunContext[BusinessDeps]) -> str:
    return "USD"

def add_context(ctx: RunContext[BusinessDeps], messages):
    return [*messages, ModelRequest(parts=[UserPromptPart(f"Policy for {ctx.deps.request_id}")])]

assistant = AssistantDefinition(
    tools=[Tool(business_tool, metadata={"owner": "host"})],
    toolsets=[toolset],
    capabilities=[ProcessHistory(add_context)],
    deps_type=BusinessDeps,
    deps_factory=dependencies,
)
"""


@pytest.fixture
def host_app(tmp_path, monkeypatch):
    path = tmp_path / "example_host.py"
    path.write_text(HOST_MODULE)
    spec = importlib.util.spec_from_file_location("example_host", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("transport", ["http", "in-process"])
async def test_external_module_uses_native_extensions_and_request_scoped_deps(
    transport, host_app, isolated_services, monkeypatch
):
    observed_policies = []
    observed_metadata = []

    async def model_response(messages, info):
        tools = {tool.name: tool for tool in info.function_tools}
        assert set(tools) == {"business_tool", "currency"}
        observed_metadata.append(tools["business_tool"].metadata)
        prompts = [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        observed_policies.append([p for p in prompts if p.startswith("Policy for ")])
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returns:
            yield {
                0: DeltaToolCall("business_tool", '{"item":"sample"}', tool_call_id="business-1"),
                1: DeltaToolCall("currency", "{}", tool_call_id="currency-1"),
            }
        else:
            yield " / ".join(str(p.content) for p in returns)

    model = FunctionModel(stream_function=model_response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    requests = [
        AssistantRequest(id=f"request-{i}", session_id=f"session-{i}", content="Look up sample")
        for i in range(2)
    ]
    if transport == "http":
        app = create_app(assistant=host_app.assistant, settings=isolated_services)
        async with app.router.lifespan_context(app):
            assert app.state.settings is isolated_services
            for service, config in [
                ("database_service", "database"),
                ("oauth_service", "oauth"),
                ("llm_service", "llm"),
                ("history_service", "history"),
                ("media_service", "media"),
                ("artifact_service", "artifacts"),
                ("tool_service", "tools"),
                ("assistant_service", "assistant"),
                ("streaming_service", "streaming"),
                ("heartbeat_service", "heartbeat"),
            ]:
                assert getattr(app.state, service)._config is getattr(isolated_services, config)
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
                responses = await asyncio.gather(
                    *[client.post("/api/chat", json=r.model_dump()) for r in requests]
                )
            assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
            contents = [r.json()["content"] for r in responses]
    else:
        async with create_runtime(
            assistant=host_app.assistant, settings=isolated_services
        ) as runtime:
            responses = await asyncio.gather(*[runtime.run_message(r) for r in requests])
            contents = [r.content for r in responses]
        assert (await runtime.health_check())["healthy"] is False

    assert contents == ["request-0:sample / USD", "request-1:sample / USD"]
    assert len(host_app.created_deps) == len(host_app.seen_deps) == 2
    assert {id(d) for d in host_app.created_deps} == {id(d) for d in host_app.seen_deps}
    assert len({id(d) for d in host_app.created_deps}) == 2
    assert observed_metadata
    assert all(m == {"owner": "host"} for m in observed_metadata)
    assert all(policies for policies in observed_policies)
    assert {p for policies in observed_policies for p in policies} == {
        "Policy for request-0",
        "Policy for request-1",
    }


async def test_runtime_closes_services_when_host_code_raises(isolated_services):
    async def use_runtime():
        async with create_runtime(settings=isolated_services) as runtime:
            observed.append(runtime)
            raise ValueError("Host operation failed")

    observed = []
    with pytest.raises(ValueError, match="Host operation failed"):
        await use_runtime()
    assert (await observed[0].health_check())["healthy"] is False


@pytest.mark.parametrize("cancel", [False, True], ids=["factory-error", "factory-cancelled"])
async def test_dependency_failure_does_not_leave_session_marked_in_flight(
    isolated_services, cancel
):
    started, release = asyncio.Event(), asyncio.Event()

    async def dependencies(request):
        started.set()
        await release.wait()
        raise RuntimeError("Dependency unavailable")

    definition = AssistantDefinition(deps_factory=dependencies)
    async with create_runtime(assistant=definition, settings=isolated_services) as runtime:
        task = asyncio.create_task(
            runtime.run_message(
                AssistantRequest(id="user-1", session_id="session-1", content="Help")
            )
        )
        try:
            async with asyncio.timeout(5):
                await started.wait()
                if cancel:
                    task.cancel()
                else:
                    release.set()
                with pytest.raises(asyncio.CancelledError if cancel else AgentRunError):
                    await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        # A failed setup must not queue future steering behind a nonexistent run.
        status = await runtime.accept_steering(
            AssistantRequest(
                id="steering-1",
                session_id="session-1",
                content="Try again",
                message_type="steering",
            ),
            has_live_stream=False,
        )
        assert status == "promoted"


def test_socket_application_passes_the_same_definition_and_settings(host_app, isolated_services):
    app = create_asgi_app(assistant=host_app.assistant, settings=isolated_services)
    fastapi_app = app.other_asgi_app
    assert fastapi_app.state.assistant_definition is host_app.assistant
    assert fastapi_app.state.settings is isolated_services
    assert fastapi_app.state.sio.fastapi_app is fastapi_app


async def test_selected_builtins_and_provider_tools_keep_host_metadata(host_schema):
    service = ToolService(
        ToolConfig(
            builtin_tools=frozenset({"time"}),
            provider_capabilities=frozenset(),
            host_tools=host_schema,
            invalidations={"get_time": ["clock"]},
        ),
        providers={"notes": object()},
    )
    await service.start()
    try:
        available = service.get_available_tools()
        assert [t.name for t in available.backend_tools] == ["get_time"]
        assert [t.name for t in available.host_tools] == ["select_item"]
        assert service.is_host_tool("select_item")
        assert service.get_tool_invalidates("get_time") == ["clock"]
    finally:
        await service.stop()


async def test_unknown_provider_selection_fails_at_startup():
    service = ToolService(ToolConfig(provider_capabilities={"misspelled"}))
    with pytest.raises(ToolError, match="Unknown provider capabilities: misspelled"):
        await service.start()


def test_unknown_builtin_selection_is_rejected():
    with pytest.raises(ValidationError):
        ToolConfig(builtin_tools={"misspelled"})


async def test_two_definitions_run_their_own_profiles_and_evolve_artifacts(
    isolated_services, monkeypatch
):
    """Distinct instructions without core edits; an allowed self-edit reaches the next prompt."""
    prompts: list[str] = []

    async def model_response(messages, info):
        system = [p.content for m in messages for p in m.parts if isinstance(p, SystemPromptPart)]
        instructions = [m.instructions for m in messages if getattr(m, "instructions", None)]
        prompts.append("\n".join([*system, *instructions]))
        returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
        prompt = [p.content for m in messages for p in m.parts if isinstance(p, UserPromptPart)]
        if returns:
            yield "Noted."
        elif (
            "manage_artifacts" in {t.name for t in info.function_tools}
            and prompt[-1] == "Remember blue"
        ):
            yield {
                0: DeltaToolCall(
                    "manage_artifacts",
                    '{"action":"update","name":"scratchpad","content":"Customer prefers blue"}',
                    tool_call_id="artifact-1",
                )
            }
        else:
            yield "Hello."

    model = FunctionModel(stream_function=model_response)
    monkeypatch.setattr(LlmService, "_resolve_agent_model", lambda self, name: model)
    settings = isolated_services.model_copy(
        update={
            "tools": ToolConfig(
                builtin_tools=frozenset({"artifacts"}), provider_capabilities=frozenset()
            )
        }
    )
    shop = AssistantDefinition(
        profile=AssistantProfile(
            name="shop",
            artifacts=[
                ArtifactDefinition(
                    name="instructions", required=True, default="You help shoppers."
                ),
                ArtifactDefinition(
                    name="scratchpad", policy=ArtifactPolicy(assistant_edit="autonomous")
                ),
            ],
        )
    )
    clinic = AssistantDefinition(
        profile=AssistantProfile(
            name="clinic",
            artifacts=[
                ArtifactDefinition(
                    name="instructions", required=True, default="You book appointments."
                )
            ],
        )
    )

    async with create_runtime(assistant=shop, settings=settings) as runtime:
        first = await runtime.run_message(
            AssistantRequest(id="m1", session_id="s", content="Remember blue")
        )
        assert first.content == "Noted."
        second = await runtime.run_message(
            AssistantRequest(id="m2", session_id="s", parent_id=first.message_id, content="Hi")
        )
        assert second.content == "Hello."
    async with create_runtime(assistant=clinic, settings=settings) as runtime:
        await runtime.run_message(AssistantRequest(id="m1", session_id="s", content="Hi"))

    assert len(prompts) == 4
    assert prompts[0].startswith("You help shoppers.")
    assert "Customer prefers blue" not in prompts[0]
    # The write lands in the next turn's prompt, not in the same turn's later requests.
    assert "Customer prefers blue" not in prompts[1]
    assert "You help shoppers.\n\nCustomer prefers blue" in prompts[2]
    assert prompts[3].startswith("You book appointments.")
    assert "Customer prefers blue" not in prompts[3]
    for text in prompts:
        assert "operator" not in text.lower()
