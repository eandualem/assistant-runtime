# Assistant Runtime

A Python assistant backend built on [Pydantic AI](https://pydantic.dev/ai/).
It brings conversation state, application context, tools, and streaming
together behind an application you already have. It receives messages,
runs the configured agent, and streams results over Socket.IO or returns
them over HTTP. The server uses FastAPI and python-socketio, works with
Anthropic, OpenAI, Google and OpenRouter models, and integrates with
[agent-backbone](https://github.com/eandualem/agent-backbone) for
managing terminal AI agents.

What you get:

- A conversation model that survives real use: sessions as message trees,
  mid-turn steering, history compaction, working memory, and a versioned
  system prompt you edit at runtime.
- A tool system: built-in tools, capabilities (notes, a document
  library, peers, rooms, reminders, issues, messaging, ...) served by
  whichever providers you configure, MCP servers, and **host tools**:
  actions your own application performs when the model asks.
- A streaming contract designed for user interfaces: thinking, text, tool
  calls and results as ordered events, with a continuation protocol for
  tools the host executes. Cancelling a turn retains partial text and
  completed tool results in its session.
- A terminal chat, so you can try all of it with one API key and no
  separate user interface.

## Quick start

```bash
git clone https://github.com/eandualem/assistant-runtime
cd assistant-runtime
uv sync
export ANTHROPIC_API_KEY=sk-ant-...       # or OPENAI_API_KEY, GOOGLE_API_KEY, OPENROUTER_API_KEY
uv run assistant-runtime chat
```

`chat` runs the runtime in-process and streams the reply to the terminal.
`uv run assistant-runtime serve` runs the HTTP and Socket.IO server on
`127.0.0.1:7100`, replacing a previous runtime left on that port; `doctor`
reports what is configured. Postgres is
optional: without it sessions and prompt-artifact versions live in memory.
With it (`make db-up && make db-upgrade`) sessions, artifact versions,
settings and a host action waiting for its result persist across restarts
(see [docs/persistence.md](docs/persistence.md)). The assistant's artifacts (its instructions, what it may
rewrite about itself) come from a profile: the neutral built-in,
`ASSISTANT__PROFILE=technical_operator` for the example operator assistant,
a TOML file, or `AssistantDefinition(profile=...)` in host code.

The full walkthrough is in [docs/getting-started.md](docs/getting-started.md).

## Documentation

The pages ship with the package: `assistant-runtime docs` lists them and
`assistant-runtime docs <page>` prints one, so nothing here needs a
checkout.

| Page | What it covers |
|---|---|
| [concepts](docs/concepts.md) | sessions, turns, tools, host tools, host context, artifacts, envelopes |
| [host contract](docs/host-contract.md) | the versioned `host_context`, attachments and action protocol a host uses |
| [identity and access](docs/access.md) | authentication modes, session ownership, administration, CORS |
| [persistence](docs/persistence.md) | what is stored, pending host actions across restarts, action outcomes, recovery and worker topology |
| [getting-started](docs/getting-started.md) | install, one key, chat, server, a minimal client, Postgres, integrations |
| [configuration](docs/configuration.md) | every setting, the three configuration tiers, secrets |
| [api](docs/api.md) | HTTP endpoints, the Socket.IO streaming contract and the AG-UI endpoint |
| [composition](docs/composition.md) | native tools, capabilities and dependencies in a host-owned Python application |
| [compatibility](docs/compatibility.md) | tested Pydantic AI versions and migration boundaries |

## Putting it behind your application

For a Python host, pass an `AssistantDefinition` with native Pydantic AI
tools, toolsets, capabilities and a dependency factory to `create_asgi_app`
or `create_runtime`. Both accept the same `AppSettings` and use the same
turn pipeline. See [composing an assistant](docs/composition.md) for a complete
server and in-process example.

Stop an active turn through `POST /api/chat/{session_id}/cancel`, Socket.IO
`assistant_cancel`, or the in-process `cancel_session` method. Cancellation
saves the work completed so far and drains running tools before the next
turn starts. A Socket.IO disconnect leaves the turn running; reconnect and
read the session to recover its saved state. See [turn control](docs/api.md#turn-control).

The runtime knows nothing about any particular host. Your application
describes itself in two ways, both optional:

- **Host context**, sent with a message: what the user is looking at, as
  a small versioned JSON object (`host`, `view`, `navigation`,
  `attachments`, `background`, `extensions`). It goes into the system
  prompt and can select which tools apply to the current view.
- **Host actions**, declared in configuration or per request: actions your
  application performs. The model calls them like any tool; the runtime
  emits a `tool_call` event, ends the turn with a pending call, and resumes
  when your application sends the result back.

A client needs a Socket.IO connection to the `/assistant` namespace, a
`join_session`, and a `message`. The events it receives are listed in
[docs/api.md](docs/api.md); the host contract in
[docs/host-contract.md](docs/host-contract.md).

## Configuration in one screen

Standalone commands read configuration from the environment or a `.env` file
(`.env.example` lists every variable). The essentials:

```bash
ANTHROPIC_API_KEY=sk-ant-...                     # at least one provider key
LLM__PRIMARY_MODEL=anthropic:claude-opus-5       # chat model (default)
LLM__SUMMARIZATION_MODEL=anthropic:claude-haiku-4-5

DATABASE__HOST=localhost                          # optional Postgres
DATABASE__PORT=5434

BACKBONE_URL=http://127.0.0.1:7120                # optional: agent-backbone tools
GITHUB_TOKEN=... GITHUB_REPO_OWNER=... GITHUB_REPO_NAME=...   # optional: issue tools
TELEGRAM_TOKEN=... TELEGRAM_CHAT_ID=...          # optional: Telegram tools

TOOLS__HOST_TOOLS='{"navigate": {"description": "...", "parameters": {...}}}'   # optional: host tools
TOOLS__PAGE_SCOPES='{"tasks": ["create_issue", "get_time"]}'                   # optional: page-scoped tools
```

Three tiers apply at request time: the environment, a runtime overlay
changed through `PATCH /api/settings`, and per-request overrides in the
message. Details, defaults and the optional extras (`[video]`,
`[tracing]`) are in [docs/configuration.md](docs/configuration.md).

### OpenAI through a ChatGPT/Codex subscription

For local personal use, the runtime can authenticate the same way the
Codex CLI does instead of using a usage-billed `OPENAI_API_KEY`. Set
`OAUTH__ENCRYPTION_KEY` to a Fernet key (`python -c "from
cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`),
start the server, and either run the device flow
(`POST /api/oauth/openai/device-code`, then open the returned URL and
enter the code) or import an existing Codex CLI login
(`codex login`, then `POST /api/oauth/openai/codex-cli/sync`).
`GET /api/oauth/openai/status` shows the connection;
`DELETE /api/oauth/openai` disconnects. While connected, every `openai:`
model goes through the subscription (set `LLM__CODEX_MODELS` to a JSON
list to narrow that); the backend decides what the plan allows. Other
OpenAI features, such as image generation, still use an API key.
`assistant-runtime doctor` reports the state of this path. It is not
intended for multi-user hosted services.

## Security

By default every caller is the local operator and CORS is open: bind to
localhost (the default). To serve several people, put the server behind
a reverse proxy that authenticates and sets `X-Assistant-Principal`
(`ACCESS__MODE=header`), or plug your own identity system in with
`AssistantDefinition(authenticate=...)` (`ACCESS__MODE=host`), and
restrict `ACCESS__CORS_ORIGINS`. Sessions belong to the principal that
created them; administration (settings, provider keys, artifact
mutations, ingress, debugging) needs the `admin` role. See
[identity and access](docs/access.md). Secrets are read from the
environment only; the one stored secret is the encrypted provider-key
store behind `PUT /api/providers/{provider}/api-key`.

## Development

Planned Pydantic AI reuse and assistant-framework generalization are tracked
in [#83](https://github.com/eandualem/assistant-runtime/issues/83).
The [compatibility baseline](docs/compatibility.md) records tested versions,
public API replacement boundaries, and offline regression cases. Streamed
tool arguments are preserved in session history and host continuations,
including when the provider represents them as JSON strings.

```bash
uv sync --locked --extra dev       # install runtime + test/lint tools from uv.lock
make check                        # ruff check + format check + pytest; the CI gate
make test                         # pytest only; no services needed
make dev                          # uvicorn with reload on port 7100
make db-up / db-upgrade / db-migrate MSG="..."   # Postgres in Docker, migrations
```

Python 3.12 or 3.13. Tests run without Postgres or network access.
[AGENTS.md](AGENTS.md) documents the module skeleton, the startup order,
the layering and the invariants a change must keep. Pull requests target
`develop`; CodeRabbit reviews every one.

### Working with coding agents

[AGENTS.md](AGENTS.md) is the shared source of project instructions. Codex
loads it automatically; [CLAUDE.md](CLAUDE.md) imports it for Claude Code.
With another CLI, ask it to read `AGENTS.md` before working if it does not
discover the file itself. Edit shared guidance in `AGENTS.md` so it stays
consistent across tools.

Start a new agent session from this checkout after changing the instructions.
These files provide project context; CLI credentials, permissions, MCP
connections, plugins, and private conversation memory remain configured
separately in each tool. Keep durable project knowledge in the repository
docs so future sessions can use it.

### Layout

```text
src/assistant_runtime/
  base/          lifecycle manager, protocols, resilience, exceptions
  services/      database, llm, history, tools, media, mcp, oauth, tracing
  app/           assistant (prompt, sessions), streaming (the turn pipeline), routes
  cli/           chat, serve, doctor, docs
  help/          the documentation, when installed from a wheel
  artifacts.py   assistant profiles: the artifact schema, built-ins, TOML loading
  host_context.py  the host contract: versioned context, attachments, actions
  principal.py   trusted principals, credentials, the ownership rule
  profiles/      the example technical_operator texts
  model_catalog.py  providers, key variables, fallback models, the model list
  config.py      AppSettings, composed from every module's config
  main.py        the FastAPI app, lifespan and Socket.IO wrapper
docs/            the documentation pages
alembic/         database migrations
```

## License

See [LICENSE](LICENSE).
