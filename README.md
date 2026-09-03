# Lovely Assistant

AI assistant backend for the **Lovely Console** — an agent operations dashboard. Receives chat messages, streams responses via Socket.IO, and routes tool calls to backend systems (tmux sessions, agent state, GitHub issues, backbone API, etc.).

Built with **FastAPI** + **Pydantic AI** + **Socket.IO**. Follows the backend module pattern — every capability is a module with the same 5-file skeleton (`config.py`, `deps.py`, `factory.py`, `interface.py`, `exceptions.py`).

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Start Postgres
make db-up

# 3. Configure environment
cp .env.example .env
# Edit .env — at minimum set one LLM provider API key

# 4. Run migrations
make db-upgrade

# 5. Start dev server
make dev
# Server runs on http://localhost:7100
```

## Prerequisites

- **Python 3.12+**
- **uv** (package manager)
- **Docker** (for Postgres)

## Configuration

Copy `.env.example` to `.env` and configure. The app uses Pydantic Settings with `__` as the nested delimiter (e.g., `DATABASE__HOST`).

### LLM Providers

At least one usable LLM auth path is required. That can be a direct provider API key, or OpenAI ChatGPT/Codex OAuth for supported OpenAI chat models. The assistant uses the configured primary model for chat and a separate model for summarization.

```bash
# Direct API keys (set any combination)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Model selection (optional — defaults to anthropic:claude-haiku-4-5)
LLM__PRIMARY_MODEL=anthropic:claude-sonnet-4-6
LLM__SUMMARIZATION_MODEL=anthropic:claude-haiku-4-5
```

### OpenAI Subscription (OAuth)

If you use ChatGPT/Codex sign-in locally, Jarvis can reuse that auth path instead of requiring a separate usage-billed `OPENAI_API_KEY`. The assistant authenticates via the same OAuth client the Codex CLI uses and talks to the ChatGPT/Codex backend directly for supported OpenAI chat models.

This is a local personal-use integration. It is not a generic API-key conversion layer and is not intended for multi-user hosted service usage.

**There are two ways to use OpenAI models:**

1. **Direct API key** — Set `OPENAI_API_KEY` in `.env`. Pay-per-token. Simple.
2. **OAuth subscription** — Sign in with your ChatGPT/Codex account for local personal use. One-time setup below.

#### OAuth Setup

**Recommended for local use:** start device auth from Jarvis, then complete the browser step with your ChatGPT/Codex account.

```bash
# Initiate device auth
curl -X POST http://localhost:7100/api/oauth/openai/device-code
```

Response:

```json
{
  "user_code": "ABCD-1234",
  "verification_uri": "https://auth.openai.com/codex/device",
  "expires_in": 900
}
```

Open `verification_uri`, enter `user_code`, and Jarvis will poll the OpenAI device-auth endpoint, exchange the returned authorization code, and store the refreshed OAuth chain locally.

#### Optional fallback: sync from an existing Codex CLI session

```bash
codex login
curl -X POST http://localhost:7100/api/oauth/openai/codex-cli/sync
```

This path imports the local Codex auth state from `~/.codex/auth.json` and refreshes it for Jarvis when needed.

**Step 1: Generate an encryption key** (protects tokens stored in the database):

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Step 2: Add to `.env`:**

```bash
OAUTH__ENCRYPTION_KEY=<the-key-from-step-1>
```

That's the only configuration needed. No client ID, no registration.

**Step 3: Restart the server**, then authenticate using one of the flows above.

**Step 4: Verify** — Check the connection status:

```bash
curl http://localhost:7100/api/oauth/openai/status

# Response:
# {
#   "connected": true,
#   "status": "authorized",
#   "source": "codex_cli",
#   "email": "you@example.com",
#   "api_key_preview": null,
#   "expires_at": 1742...
# }
```

Once authorized, supported OpenAI chat models use the synced ChatGPT/Codex auth session automatically. The access token refreshes automatically before chat requests. No further action is needed after the initial sign-in.

#### Supported behavior and current limitations

- OAuth-backed OpenAI chat currently supports `openai:gpt-5.4` and `openai:gpt-5.4-pro`.
- This path uses the ChatGPT/Codex backend directly. `api_key_preview` stays `null` because no reusable OpenAI API key is created.
- OpenAI image generation and other non-Codex OpenAI API features still require a direct `OPENAI_API_KEY`.
- For `openai:gpt-5.4`, `thinking_budget` maps onto OpenAI reasoning effort tiers (`low`, `medium`, `high`, `xhigh`) and requests a detailed reasoning summary.
- For `openai:gpt-5.4-pro`, enabling `thinking_budget` requests OpenAI reasoning summaries with fixed high reasoning effort.
- OpenAI/Codex exposes reasoning summaries rather than raw chain-of-thought, so Jarvis's thinking panel shows the returned summary text when available.
- If both OAuth and a direct `OPENAI_API_KEY` are configured, Jarvis uses OAuth for the supported Codex-backed chat models above. Other OpenAI API features continue to use the direct API key path.

#### OAuth Management

```bash
# Check status
curl http://localhost:7100/api/oauth/openai/status

# Disconnect (reverts to direct API key if set, or disables OpenAI)
curl -X DELETE http://localhost:7100/api/oauth/openai
```

### Database

```bash
DATABASE__HOST=localhost
DATABASE__PORT=5434
DATABASE__USER=lovely_assistant
DATABASE__PASSWORD=lovely_assistant
DATABASE__NAME=lovely_assistant
```

Default port is 5434 (mapped from Postgres container's 5432 to avoid host collisions).

### Other Services

```bash
# GitHub API (required for issue management tools)
GITHUB_TOKEN=ghp_...

# Backbone API (required for agent messaging and meeting tools)
BACKBONE_URL=http://127.0.0.1:7120

# Image/video generation (optional)
RUNWAYML_API_SECRET=...
LUMAAI_API_KEY=...

# Telegram (optional — for cross-channel messaging)
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...

# Langfuse tracing (optional)
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Development

```bash
make install       # Install dependencies (uv sync)
make dev           # Start dev server (uvicorn --reload, port 7100)
make lint          # Ruff check
make format        # Ruff format
make fix           # Auto-fix lint + format
make test          # Run all tests (pytest)
make test-file FILE=tests/test_foo.py  # Single file
make check         # Full CI gate (lint + format check + test)
make clean         # Remove build artifacts
```

### Database Commands

```bash
make db-up                    # Start Postgres (docker compose)
make db-down                  # Stop Postgres
make db-upgrade               # Run pending Alembic migrations
make db-migrate MSG="description"  # Create new migration
```

## Architecture

```
src/lovely_assistant/
├── base/               # Foundation: lifecycle, resilience, instrumentation, errors
├── services/
│   ├── database/       # SQLAlchemy + Alembic, repositories
│   ├── llm/            # Pydantic AI agent factory, model settings
│   ├── history/        # Conversation history + summarization + working memory
│   ├── tools/          # Tool registry + 15 tool domains (agent, GitHub, backbone, etc.)
│   ├── media/          # Image/video generation (OpenAI, Google, Runway, Luma)
│   ├── mcp/            # MCP server integration
│   ├── oauth/          # OpenAI OAuth, Codex CLI sync, and token management
│   └── tracing.py      # Langfuse integration
├── app/
│   ├── assistant/      # Conversation orchestration, prompt building, session store
│   ├── streaming/      # Socket.IO event streaming coordinator
│   ├── routes/         # HTTP endpoints (chat, sessions, models, settings, oauth, etc.)
│   ├── models_registry.py  # Static model catalog (40+ models across 6 providers)
│   └── settings.py     # RuntimeSettings (DB-backed user preferences)
├── config.py           # Root AppSettings (composes all module configs)
└── main.py             # FastAPI app, lifespan, Socket.IO ASGI wrapper
```

Every module follows the **5-file skeleton**: `config.py` (settings), `deps.py` (FastAPI Depends), `factory.py` (lifecycle registration), `interface.py` (public service), `exceptions.py` (error hierarchy). Modules compose via Protocol boundaries and FastAPI's dependency injection.

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Aggregated health from all lifecycle components |
| `/api/chat` | POST | Send a message, get a response (non-streaming) |
| `/api/sessions` | GET | List active sessions |
| `/api/sessions/{id}` | GET/DELETE | Get or delete a session |
| `/api/models` | GET | Model catalog, provider status, defaults, OAuth status |
| `/api/settings` | GET/PUT | User preferences (model, thinking budget, temperature) |
| `/api/artifacts` | GET/POST/PUT | Versioned prompt artifacts (system prompt fragments) |
| `/api/inbox` | GET/POST | Agent-to-assistant message inbox |
| `/api/media/*` | POST/GET | Image/video generation and job tracking |
| `/api/debug/traces` | GET | Request traces for debugging |
| `/api/oauth/openai/codex-cli/sync` | POST | Sync OAuth from local Codex CLI auth |
| `/api/oauth/openai/device-code` | POST | Initiate OAuth Device Code flow |
| `/api/oauth/openai/status` | GET | OAuth connection status |
| `/api/oauth/openai` | DELETE | Disconnect OAuth |

Streaming responses are delivered via **Socket.IO** on the `/assistant` namespace with `assistant:` prefixed events (`text_delta`, `thinking_delta`, `tool_call`, `tool_result`, `final_response`, `status`, `error`, `debug`).

## Prompt Artifacts

Jarvis's system prompt is assembled from five first-class artifacts with distinct roles:

- `soul`: enduring purpose, values, non-negotiables, deepest identity guidance
- `persona`: style, stance, behavioral voice
- `communication_protocol`: interaction and routing rules
- `ecosystem`: world model, roles, org structure, system topology
- `scratchpad`: short-lived operational memory

`Soul`, `persona`, `communication_protocol`, and `ecosystem` are durable versioned artifacts. `Scratchpad` remains the operational special case: mutable, auto-approved, and intentionally short-lived.

## Port

Default: **7100**. Override with `PORT` env var:

```bash
PORT=7101 make dev
```
