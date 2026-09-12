# Getting started

Ten minutes from nothing to a streamed conversation. Only step 2 needs
anything from you; everything else is optional.

## 0. Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- One provider API key: Anthropic, OpenAI, Google or OpenRouter
- Optional: Docker, for Postgres

## 1. Install

From a checkout:

```bash
git clone https://github.com/eandualem/assistant-runtime
cd assistant-runtime
uv sync                     # add --extra video --extra tracing for the optional features
```

Or install the released command-line tool: `uv tool install assistant-runtime`.
The sections below show commands for each installation method; use the one
that matches yours. To try a complete browser application, see the
[reference-app guide](reference-app.md).

## 2. One key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

From a checkout:

```bash
uv run assistant-runtime doctor
```

With the installed tool:

```bash
assistant-runtime doctor
```

`doctor` prints one line per check: Python version, `.env`, provider keys,
the model ids in use, whether Postgres is reachable, which extras are
installed. It exits non-zero when no provider key is set. A `.env` file in
the working directory is read too; `.env.example` lists every variable.

Any one provider key is enough. The default chat model is
`anthropic:claude-opus-5`; with only an OpenAI, Google or OpenRouter key
the runtime uses that provider's default instead (`openai:gpt-5.6-terra`,
`google:gemini-3.1-pro-preview`, `openrouter:x-ai/grok-4.1-fast`). Set
`LLM__PRIMARY_MODEL=provider:name` to choose.

## 3. Talk to it

From a checkout:

```bash
uv run assistant-runtime chat
```

With the installed tool:

```bash
assistant-runtime chat
```

This runs the runtime in-process, with the same services, tools and prompt
as the server, and streams the reply to your terminal. Tool calls show on
their own lines. `--show-thinking` prints the model's thinking,
`--model openai:gpt-5.6-terra` switches models for this chat,
`-m "one message"` sends a single message and exits, `/exit` quits.

Without Postgres the session lives in memory and the bundled default prompt
artifacts are used. Tools that need an integration you have not configured
(GitHub, Telegram, agent-backbone) answer with a structured error and the
model carries on.

## 4. Run the server

From a checkout:

```bash
uv run assistant-runtime serve            # 127.0.0.1:7100
```

With the installed tool:

```bash
assistant-runtime serve                   # 127.0.0.1:7100
```

From another terminal, check the server and send a message:

```bash
curl -s localhost:7100/health
curl -s -X POST localhost:7100/api/chat -H 'content-type: application/json' \
  -d '{"id":"m1","session_id":"s1","content":"What can you do?"}'
```

`serve` binds to loopback by default. The API has no authentication and
CORS is open, so keep it on localhost or put it behind your own reverse
proxy with auth. `--host 0.0.0.0 --port 8080` changes the binding,
`--reload` restarts on source changes (`make dev` is the same thing).
If a previous assistant-runtime still holds the port (recognised by its
`/health` answer), `serve` stops it and takes over, so a new configuration
takes effect with one command; anything else on the port is left alone and
reported. `--no-replace` turns the takeover off.

## 5. Stream from a client

Streaming goes over Socket.IO, namespace `/assistant`. The smallest client:

```python
import socketio, uuid

sio = socketio.Client()

@sio.on("assistant:text_delta", namespace="/assistant")
def on_text(event):
    print(event["content"], end="", flush=True)

@sio.on("assistant:status", namespace="/assistant")
def on_status(event):
    if event["status"] == "completed":
        print()
        sio.disconnect()

sio.connect("http://127.0.0.1:7100", namespaces=["/assistant"])
sio.emit("assistant_join_session", {"session_id": "s1"}, namespace="/assistant")
sio.emit("assistant_message",
         {"id": str(uuid.uuid4()), "session_id": "s1", "content": "Hello"},
         namespace="/assistant")
sio.wait()
```

Every event the runtime emits is listed in [api](api.md).

## 6. Optional: Postgres

Persistent sessions, editable and versioned prompt artifacts, runtime
settings that survive a restart, the inbox and the heartbeat:

```bash
make db-up          # docker compose: Postgres on localhost:5434
make db-upgrade     # alembic migrations
uv run assistant-runtime serve
```

From an installed package there is no Makefile, so create the schema with
the runtime itself; the migrations ship inside the wheel:

```bash
assistant-runtime migrate     # alembic upgrade head, using DATABASE__*
```

Connection settings are `DATABASE__HOST`, `DATABASE__PORT`,
`DATABASE__USER`, `DATABASE__PASSWORD`, `DATABASE__NAME`. When Postgres is
unreachable at startup the runtime logs a warning and runs in memory-only
mode; database-backed endpoints return 503. This is also the local
persistent option: what survives a restart, and how a host action waiting
for its result is recovered, is in [persistence](persistence.md).

## 7. Optional: an AG-UI frontend

Install the `ag-ui` extra (`uv sync --extra ag-ui`, or
`pip install "assistant-runtime[ag-ui]"`) and point an AG-UI client at
`POST /api/agui`; `examples/agui/index.html` is a dependency-free page that
does it with `fetch`. The mapping to sessions and host actions is in
[api](api.md#ag-ui-post-apiagui).

## 8. Optional: describe your host

If you are putting the runtime behind your own application, send
`host_context` with your messages and, if the application can perform
actions on the model's behalf, declare them as host tools. The shape and
the settings are in [concepts](concepts.md) and
[configuration](configuration.md).

## 9. Optional: integrations

| To use | Set |
|---|---|
| agent-backbone tools (agents, meetings, schedule, swarms, telemetry) | `BACKBONE_URL`, optionally `BACKBONE_API_KEY` |
| GitHub issue tools | `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME` |
| Telegram tools | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` |
| MCP servers | copy `mcp_servers.example.json` to `mcp_servers.json`, or set `MCP_CONFIG_PATH` |
| Langfuse tracing | the `[tracing]` extra and `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` |
| Video generation | the `[video]` extra and `RUNWAYML_API_SECRET` or `LUMAAI_API_KEY` |
| OpenAI through a ChatGPT/Codex subscription instead of an API key | `OAUTH__ENCRYPTION_KEY`, then `POST /api/oauth/openai/device-code` |

## Where things are

- `.env` in the working directory: keys and settings (never committed)
- `mcp_servers.json` in the working directory: MCP servers (gitignored)
- `src/assistant_runtime/artifacts.py`: the built-in assistant profiles
- `src/assistant_runtime/profiles/`: the example profile texts
- `alembic/versions/`: database migrations
- `assistant-runtime docs <page>`: this documentation, from the installed package
