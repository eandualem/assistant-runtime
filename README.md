# Assistant Runtime

An assistant backend that plugs into any work environment. It receives
messages, runs a model with a set of tools, and streams the result back
over Socket.IO or returns it over HTTP. It is built on FastAPI,
python-socketio and pydantic-ai, works with Anthropic, OpenAI, Google and
OpenRouter models, and integrates with
[agent-backbone](https://github.com/eandualem/agent-backbone) for
managing terminal AI agents.

What you get:

- A conversation model that survives real use: sessions as message trees,
  mid-turn steering, history compaction, working memory, and a versioned
  system prompt you edit at runtime.
- A tool system with backend tools (agents, GitHub issues, Telegram,
  meetings, schedules, image generation, MCP servers) and **host tools**:
  actions your own application performs when the model asks.
- A streaming contract designed for user interfaces: thinking, text, tool
  calls and results as ordered events, with a continuation protocol for
  tools the host executes.
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
`127.0.0.1:7100`; `doctor` reports what is configured. Postgres is
optional: without it sessions live in memory and the bundled prompt
artifacts are used. With it (`make db-up && make db-upgrade`) sessions
persist and artifacts and settings become editable through the API.

The full walkthrough is in [docs/getting-started.md](docs/getting-started.md).

## Documentation

The pages ship with the package: `assistant-runtime docs` lists them and
`assistant-runtime docs <page>` prints one, so nothing here needs a
checkout.

| Page | What it covers |
|---|---|
| [concepts](docs/concepts.md) | sessions, turns, tools, host tools, host context, artifacts, envelopes |
| [getting-started](docs/getting-started.md) | install, one key, chat, server, a minimal client, Postgres, integrations |
| [configuration](docs/configuration.md) | every setting, the three configuration tiers, secrets |
| [api](docs/api.md) | HTTP endpoints and the Socket.IO streaming contract |

## Putting it behind your application

The runtime knows nothing about any particular host. Your application
describes itself in two ways, both optional:

- **Host context**, sent with a message: what the user is looking at, as
  a small JSON object (`page`, `navigation`, `background`). It goes into
  the system prompt and can select which tools apply to the current page.
- **Host tools**, declared in configuration: actions your application
  performs. The model calls them like any tool; the runtime emits a
  `tool_call` event, ends the turn with a pending call, and resumes when
  your application sends the result back.

A client needs a Socket.IO connection to the `/assistant` namespace, a
`join_session`, and a `message`. The events it receives are listed in
[docs/api.md](docs/api.md); the host contract in
[docs/concepts.md](docs/concepts.md).

## Configuration in one screen

Everything comes from the environment or a `.env` file
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
`DELETE /api/oauth/openai` disconnects. This path serves the OpenAI chat
models the Codex backend supports; other OpenAI features still use an
API key. It is not intended for multi-user hosted services.

## Security

There is no authentication and CORS is open. Bind to localhost (the
default) or put the server behind a reverse proxy that authenticates.
Secrets are read from the environment only; the one stored secret is the
encrypted provider-key store behind `PUT /api/providers/{provider}/api-key`.

## Development

```bash
make check                        # ruff check + format check + pytest; the CI gate
make test                         # pytest only; no services needed
make dev                          # uvicorn with reload on port 7100
make db-up / db-upgrade / db-migrate MSG="..."   # Postgres in Docker, migrations
```

Python 3.12 or 3.13. Tests run without Postgres or network access.
`CLAUDE.md` documents the module skeleton, the startup order, the layering
and the invariants a change must keep. Pull requests target `develop`;
CodeRabbit reviews every one.

### Layout

```text
src/assistant_runtime/
  base/          lifecycle manager, protocols, resilience, exceptions
  services/      database, llm, history, tools, media, mcp, oauth, tracing
  app/           assistant (prompt, sessions), streaming (the turn pipeline), routes
  cli/           chat, serve, doctor, docs
  help/          the documentation, when installed from a wheel
  artifacts.py   the prompt artifact catalog
  config.py      AppSettings, composed from every module's config
  main.py        the FastAPI app, lifespan and Socket.IO wrapper
docs/            the documentation pages
alembic/         database migrations
```

## License

See [LICENSE](LICENSE).
