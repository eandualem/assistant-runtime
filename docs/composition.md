# Composing an assistant

Pass an `AssistantDefinition` and optional `AppSettings` to `create_app`
(HTTP), `create_asgi_app` (HTTP + Socket.IO), or `create_runtime`
(in-process). All three use the same service factories and turn pipeline.
The standalone CLI/server still works with no definition.

An assistant definition accepts native Pydantic AI `Tool` objects or tool
functions, `AbstractToolset` instances, `AgentCapability` objects/functions,
a `deps_type`, and a sync or async `deps_factory`. It snapshots the supplied
sequences into tuples and passes the extension objects to Pydantic AI.
It does not replace their schemas, validation, preparation or retry behavior.
Execution uses Pydantic AI's public event stream, so native node guards and
middleware run around the actual model/tool execution. Native tool outputs
are converted to JSON for session storage, with media omitted under the
runtime's existing retention policy. Failed native tool outcomes remain
failures in client events and restored conversation history.

## A host-owned Python module

Save this as `store_assistant.py` in your application. Configure a model
provider as in [getting started](getting-started.md).

```python
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai.capabilities import PrepareTools
from pydantic_ai.tools import ToolDefinition

from assistant_runtime.artifacts import ArtifactDefinition, ArtifactPolicy, AssistantProfile
from assistant_runtime.config import AppSettings
from assistant_runtime.main import AssistantDefinition, create_asgi_app
from assistant_runtime.services.tools.config import ToolConfig
from assistant_runtime.services.tools.providers.config import ProvidersConfig


@dataclass
class StoreDeps:
    catalog: dict[str, str]


def dependencies(request) -> StoreDeps:
    # A real host can resolve request resources through its own services.
    return StoreDeps(catalog={"lamp": "Desk lamp, warm white light"})


def product_details(ctx: RunContext[StoreDeps], sku: str) -> str:
    """Look up a product in the current catalog."""
    return ctx.deps.catalog.get(sku, "Product not found")


def available_tools(
    ctx: RunContext[StoreDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition]:
    # Hide the catalog tool while the host has no catalog available.
    return [
        tool for tool in tool_defs
        if tool.name != "product_details" or ctx.deps.catalog
    ]


profile = AssistantProfile(
    name="store",
    artifacts=[
        ArtifactDefinition(
            name="instructions",
            role="what the assistant is for",
            required=True,
            default="You help customers of the store find and compare products.",
        ),
        ArtifactDefinition(
            name="tone",
            role="voice and style",
            default="Friendly and brief.",
            policy=ArtifactPolicy(assistant_edit="propose"),
        ),
        ArtifactDefinition(
            name="scratchpad",
            role="notes the assistant keeps for itself",
            policy=ArtifactPolicy(assistant_edit="autonomous"),
        ),
    ],
)

assistant = AssistantDefinition(
    tools=[product_details],
    capabilities=[PrepareTools(available_tools)],
    deps_type=StoreDeps,
    deps_factory=dependencies,
    profile=profile,
)
settings = AppSettings(
    # "artifacts" registers manage_artifacts, so the model can evolve its scratchpad.
    tools=ToolConfig(builtin_tools=frozenset({"artifacts"}), provider_capabilities=frozenset()),
    providers=ProvidersConfig(),
)
app = create_asgi_app(assistant=assistant, settings=settings)
```

Run `uvicorn store_assistant:app --host 127.0.0.1 --port 7100` from your host
environment. The existing HTTP and Socket.IO contracts apply. You can mount
the ASGI application in a host server; that host must also manage the
runtime application's lifespan. `create_app` returns the underlying FastAPI
application for hosts that want HTTP only.

The same definition works without a listening server:

```python
import asyncio

from assistant_runtime.app.assistant.models import AssistantRequest
from assistant_runtime.main import create_runtime
from store_assistant import assistant, settings


async def main():
    async with create_runtime(assistant=assistant, settings=settings) as runtime:
        result = await runtime.run_message(AssistantRequest(
            id="message-1", session_id="conversation-1", content="Tell me about the lamp"
        ))
        print(result.content)


asyncio.run(main())
```

`runtime.stream_message(request)` yields the same event dicts as the server's
turn pipeline. Follow-up requests use the returned assistant message ID as
`parent_id`; host-tool continuations use the existing matching call-ID
contract. The context manager shuts services down if host code raises or
startup is interrupted.

## The assistant's artifacts

`profile` defines the prompt's leading fragments: their names, order,
default text and mutation policy. It replaces the neutral built-in and the
`ASSISTANT__PROFILE` setting. The same profile can be a TOML file instead
(see `load_profile_file` in `assistant_runtime.artifacts`). Stored
versions are scoped by the profile's `name`: two hosts with different
names never share them, and the built-ins use `neutral` and
`technical_operator` (rows written before profiles existed belong to the
latter). A host that wants the original technical-operator assistant uses
`technical_operator_profile()`.

Policies are enforced by the runtime, so the example above lets the model
rewrite its `scratchpad` immediately, propose a new `tone` for the host to
approve through `/api/artifacts`, and only read `instructions` until the
host changes it. See [concepts](concepts.md#prompt-artifacts-and-profiles).

## Identity

With `ACCESS__MODE=host`, `AssistantDefinition(authenticate=...)` turns a
transport's `Credentials` (headers, the Socket.IO connect payload, the
client address) into a `Principal`, or None to reject the caller. Sessions
belong to the principal that created them; in-process callers pass
`principal=` to `run_message` and friends, or act as the local operator.
See [identity and access](access.md).

## Cancelling a turn

The runtime uses Pydantic AI's `CancellationToken` and `RunCancelled`
snapshots to retain partial text, completed tool results and available usage.
From a host stop handler, while another task consumes the turn:

```python
cancel_requested = await runtime.cancel_session("conversation-1")
await runtime.wait_for_session("conversation-1")
```

`cancel_session` reports whether it found a cancellable turn; it also stops
dependency setup. `wait_for_session` waits for the registered turn's
persistence and cleanup. In-process `run_message` raises `AgentRunError`
for a cancelled turn; `stream_message` delivers the cancellation's
final/error/completed events. Saved partial work remains in the session.

Closing or cancelling the event iterator requests cancellation and drains
the producer. Use `contextlib.aclosing` when breaking out early. External
`asyncio.CancelledError` still propagates to the caller. Runtime shutdown
cancels and drains active turns before stopping their services. Socket.IO
disconnects have a different policy: the server keeps the turn running.

An unresolved tool is recorded as interrupted, with its external outcome
unknown. Cancellation cannot undo side effects or revoke an idle host action.
Steering is acknowledged after a successful model response consumes it; if
that response is interrupted, the instruction stays pending for the next
turn. This retry behavior is not an exactly-once side-effect guarantee.

## Dependencies and lifetime

The dependency factory receives the validated `AssistantRequest` once per
accepted turn or host-tool continuation. Its return value becomes
`RunContext.deps` for tools and native capabilities throughout that turn,
including any additional agent run used to deliver queued steering. An
async factory is awaited before agent setup. Dependencies are not persisted
in conversation history and are freshly resolved on continuation.

The host owns resources returned by the factory, including their cleanup.
Use host-managed pools or resource scopes. For HTTP, middleware or a host
dependency can establish trusted request context that the factory reads;
for Socket.IO, establish it in the host's event/task scope. A payload's
`session_id` and `host_context` are client input, not authenticated identity.
This API does not add authentication or tenant isolation.

Native tools/toolsets/capabilities may be reused across concurrent requests,
as they are in Pydantic AI. Keep per-request mutable data in dependencies or
native run state instead of mutating a shared extension object. The runtime's
existing subagent tool does not inherit host-native extensions or dependencies.

## Selecting runtime tools

`AppSettings` is resolved once at startup and the supplied object is passed
to every settings-based service factory. Without an explicit object, the
environment and `.env` provide settings as before. Runtime and per-request
tunable overrides still apply above this startup configuration.

- `tools.builtin_tools`: enabled groups from `time`, `screen`, `artifacts`,
  `subagent`, `media`, `video`. All are selected by default; an empty set
  selects none. Existing service requirements still apply.
- `tools.provider_capabilities`: enabled business capabilities from
  configured providers. `None` keeps every configured capability; an empty
  set selects none. Names are validated at startup. A selection never
  enables an unconfigured provider.
- `providers=ProvidersConfig()` disables automatic provider discovery for
  that settings object. The default still discovers configured providers
  from their existing environment variables. Provider credentials and MCP
  configuration keep their documented configuration paths.

These selections do not suppress explicitly configured host tools, MCP
servers, or native extensions. Configure those separately. Existing runtime
tool descriptions, page scopes, invalidation domains and host categories
retain their behavior. Native tools keep their Pydantic AI metadata and
preparation hooks; they are not inserted into the runtime's static provider
catalog or filtered by `tools.page_scopes`. Use native `prepare` functions
or `PrepareTools` for dynamic selection. `tools.invalidations` can name
native tools because event invalidation lookup is by tool name.

Two different uses of “capability” exist in the code:

| Name | Meaning | Extension point |
|---|---|---|
| Runtime business capability | A group of integration tools, such as notes or issues, served by a configured provider | `AppSettings.providers`, `tools.provider_capabilities` |
| Pydantic AI capability | A native agent behavior extension, such as history processing or tool preparation | `AssistantDefinition.capabilities` |

The supported native contracts are documented in
[Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
and [toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/).
No optional UI, tracing, video or durable-backend package is required for
this composition API. The compatibility cases run an external module through
HTTP and in-process calls with real agent execution and concurrent dependency
resolution.

This definition supplies native extensions. Configurable prompt artifact
schemas and neutral defaults are tracked separately in #88. External toolsets
that defer execution still need the runtime's configured host-tool names and
schemas; arbitrary deferred-action protocols and native custom-event mapping
are not introduced by this API.
