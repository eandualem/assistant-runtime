# Assistant Runtime

[![PyPI](https://img.shields.io/pypi/v/assistant-runtime)](https://pypi.org/project/assistant-runtime/)
[![Downloads](https://img.shields.io/pypi/dm/assistant-runtime)](https://pypistats.org/packages/assistant-runtime)
[![CI](https://github.com/eandualem/assistant-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/eandualem/assistant-runtime/actions/workflows/ci.yml)

An assistant backend for the application you already have: it holds the conversation, understands what is on screen, and asks your app to act.

![Design Studio showing a design document, Mermaid diagram and assistant actions](https://raw.githubusercontent.com/eandualem/design-studio/main/public/screenshot.png)

*Design Studio: the assistant edits a document through actions the application provides.*
<!-- Elias records this -->
<!-- Replace the screenshot above with ![Assistant Runtime in the two studios](docs/media/demo.gif) and a recording caption. -->

**Contents** · [What it enables](#what-it-enables) · [See it in an application](#see-it-in-an-application) · [Getting started](#getting-started) · [How it works](#how-it-works) · [How it relates to other tools](#how-it-relates-to-other-tools) · [Documentation](#documentation) · [Development](#development)

## What it enables

- **A conversation people can keep working in.** Branch from an earlier message, steer a turn while it runs, retain working memory, and edit the assistant's versioned system prompt.
- **An assistant that can use your application.** Send what is on screen and declare actions such as editing a document or moving an object. The runtime requests an action; your app performs it and returns the result to continue the turn.
- **A UI that follows the work.** Thinking, text, tool calls and results arrive as ordered events. Cancellation saves partial text and completed work.
- **A second model role in parallel.** A silent controller can return one application action or a structured hold with `output_mode: host_tools`, while another model handles the conversation.
- **Voice with the same application context.** GPT-Live carries the conversation over WebRTC and can delegate tasks into the shared tool pipeline, or speak alongside an independent controller.
- **Typed decisions while the user is still speaking.** Send your application's state and a set of `choice`, `score` and `noul` questions in one call; a decision model answers all of them with calibrated probabilities in one round trip, with no text to parse and about a millisecond of runtime overhead on top of the provider's own latency ([decisions](https://github.com/eandualem/assistant-runtime/blob/main/docs/decisions.md)).

One runtime can serve several applications, each with its own registered profile and host actions.

## See it in an application

### Design Studio

[Design Studio](https://github.com/eandualem/design-studio) is a Markdown and Mermaid editor with an assistant beside the document. Describe a system by talking or typing; the assistant writes and revises the design. The screenshot above shows its distinguishing loop: the app returns diagram-rendering results with each edit, so the assistant can repair a diagram that fails to render. [Run the integration](https://github.com/eandualem/assistant-runtime/blob/main/docs/reference-app.md).

### Avatar Studio

[Avatar Studio](https://github.com/eandualem/avatar-studio) is a 3D character you can talk to. Voice and body share one runtime: GPT-Live speaks while a separate controller chooses movement, and the app applies it to the character. It demonstrates parallel model roles and physical actions with execution receipts. [Run the studio](https://github.com/eandualem/avatar-studio#running-it).

![Avatar Studio showing its 3D character, conversation and voice controls](https://raw.githubusercontent.com/eandualem/assistant-runtime/main/docs/media/avatar-studio.jpg)

*Avatar Studio after a request to wave: the app applies movement and displays the conversation.*

## Getting started

You need Python 3.12 or newer, [uv](https://docs.astral.sh/uv/), and one provider API key.

```bash
uv tool install assistant-runtime
export ANTHROPIC_API_KEY=your-api-key
assistant-runtime chat
```

This opens a terminal conversation. OpenAI, Google, OpenRouter and Cerebras keys also work; see [provider configuration](https://github.com/eandualem/assistant-runtime/blob/main/docs/configuration.md).

For a browser app, exit chat and run `assistant-runtime serve`. Point the app at `http://127.0.0.1:7100`, then start the app in another terminal. Apps connect to the backend and report if it is missing; they do not start or stop it. Follow the [Design Studio recipe](https://github.com/eandualem/assistant-runtime/blob/main/docs/reference-app.md) for a complete first integration.

Postgres is optional; it persists conversations and pending host actions, including while background memory updates run: [persistence](https://github.com/eandualem/assistant-runtime/blob/main/docs/persistence.md).
Register application profiles to share a runtime: [deployments](https://github.com/eandualem/assistant-runtime/blob/main/docs/deployments.md).
Voice needs the `[voice]` extra, an OpenAI API key and explicit enablement; each call reserves a session the caller is authorized to access: [voice setup](https://github.com/eandualem/assistant-runtime/blob/main/docs/voice.md).
Local ChatGPT/Codex subscription authentication is also supported for backend models: [subscription setup](https://github.com/eandualem/assistant-runtime/blob/main/docs/subscription.md).

The server defaults to localhost and trusts the local operator. Before exposing it to other people, configure authentication and allowed browser origins: [identity and access](https://github.com/eandualem/assistant-runtime/blob/main/docs/access.md).

## How it works

Your application sends a message, its current context and the actions it can perform. The runtime builds the assistant from its profile, calls the model through Pydantic AI, and streams the turn over Socket.IO or returns it over HTTP. An optional AG-UI endpoint exposes the same pipeline.

When the model requests a host action, the runtime saves the pending call and returns it to your application. Your app validates and executes it, then sends back the matching result. The conversation resumes from that result. The app owns its interface and side effects; the runtime owns conversation state, model execution and continuation tracking.

Add another app by supplying a profile, context and actions. No runtime code needs to know the app's name. Python hosts can also compose native tools, capabilities and dependencies through `AssistantDefinition`: [composition](https://github.com/eandualem/assistant-runtime/blob/main/docs/composition.md).

## How it relates to other tools

[Pydantic AI](https://ai.pydantic.dev/) is the agent library underneath: models, typed tools, execution and streaming. Assistant Runtime is the running service around it that your application calls, with sessions, host-action continuations, profiles and voice already connected.

[Vercel AI SDK](https://ai-sdk.dev/docs/introduction) provides TypeScript model and tool APIs plus UI bindings; [AG-UI](https://docs.ag-ui.com/introduction) defines an event protocol between agents and interfaces. They help build the client/server connection. This runtime supplies the application state and server behavior behind that connection, and offers an AG-UI adapter.

[LangGraph's Agent Server](https://docs.langchain.com/langsmith/agent-server) supplies deployment, persistent threads and runs for graphs. The [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents/sdk) supplies an agent loop, tools and orchestration inside your own application. Both can support application integrations; Assistant Runtime packages a specific profile/context/action-result contract for apps that own the actions.

A FastAPI service around Pydantic AI is the closest alternative: you choose the session model, streaming protocol, persistence and UI action lifecycle yourself. This project supplies those choices as a backend you can run or extend.

## Documentation

Start with [getting started](https://github.com/eandualem/assistant-runtime/blob/main/docs/getting-started.md), then [concepts](https://github.com/eandualem/assistant-runtime/blob/main/docs/concepts.md), the [host contract](https://github.com/eandualem/assistant-runtime/blob/main/docs/host-contract.md) and the [API](https://github.com/eandualem/assistant-runtime/blob/main/docs/api.md). The [documentation index](https://github.com/eandualem/assistant-runtime/blob/main/docs/README.md) links configuration, deployment, access, persistence and composition references.

`assistant-runtime --version` prints the installed release; the OpenAPI schema reports the same version.

The pages ship with the package: `assistant-runtime docs` lists them and `assistant-runtime docs <page>` prints one without a checkout.

## Development

```bash
git clone https://github.com/eandualem/assistant-runtime && cd assistant-runtime
uv sync --locked --extra dev
make check
make dev
```

`make check` runs Ruff and the tests without API keys, Postgres or a running server. Pull requests target `develop`; [AGENTS.md](https://github.com/eandualem/assistant-runtime/blob/main/AGENTS.md) defines the architecture and contribution checks.

```text
src/assistant_runtime/base/       lifecycle and shared protocols
src/assistant_runtime/services/  models, history, tools and integrations
src/assistant_runtime/app/       conversations, streaming, voice and routes
src/assistant_runtime/cli/       chat, serve, doctor, docs and migrations
docs/ and alembic/               documentation and database migrations
```

## License

MIT — see [LICENSE](https://github.com/eandualem/assistant-runtime/blob/main/LICENSE).
