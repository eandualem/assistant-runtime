# Getting started

Install the runtime, send a first message, then connect an application.
Postgres and the optional integrations can be added later.

## 0. Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- One provider API key: Anthropic, OpenAI, Google, OpenRouter or Cerebras
  (for local subscription authentication instead, see [subscription](subscription.md))
- Optional: Docker, for Postgres

## 1. Install

For the released command-line tool:

```bash
uv tool install assistant-runtime
```

For a source checkout:

```bash
git clone https://github.com/eandualem/assistant-runtime
cd assistant-runtime
uv sync --locked
```

The commands below use the installed tool. In a checkout, prefix each
`assistant-runtime` command with `uv run`. For contribution checks, install
the development dependencies with `uv sync --locked --extra dev`.

## 2. One key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

```bash
assistant-runtime doctor
```

`doctor` prints one line per check: Python version, `.env`, provider keys,
the model ids in use, whether Postgres is reachable, which extras are
installed. It exits non-zero when no provider key is set. A `.env` file in
the working directory is read too; `.env.example` shows common settings.
A missing optional service may produce a warning without preventing chat.
`doctor` currently reports the missing API key even when you plan to use
[subscription authentication](subscription.md); check that path separately.

Any one provider key is enough. The default chat model is
`anthropic:claude-opus-5`; with only another supported provider's key,
the runtime uses that provider's default instead. See
[provider defaults](configuration.md#providers-and-models). Set
`LLM__PRIMARY_MODEL=provider:name` to choose.

## 3. Talk to it

```bash
assistant-runtime chat
```

This runs the runtime in-process, with the same services, tools and prompt
as the server, and streams the reply to your terminal. Tool calls show on
their own lines. `--show-thinking` prints the model's thinking,
`--model openai:gpt-5.6-terra` switches models for this chat,
`-m "one message"` sends a single message and exits, `/exit` quits.
Without credentials for the chosen model (a key or a connected
subscription), `chat` stops at startup and names the keys it accepts.

Without Postgres, conversations and edits to prompt artifacts live in process
memory and are lost on restart. Integration tools such as GitHub and Telegram
are offered to the model only when their providers are configured.

## 4. Run the server

```bash
assistant-runtime serve                   # 127.0.0.1:7100
```

From another terminal, check the server and send a message:

```bash
curl -s localhost:7100/health
curl -s -X POST localhost:7100/api/chat -H 'content-type: application/json' \
  -d '{"id":"m1","session_id":"s1","content":"What can you do?"}'
```

The chat response is JSON: `content` holds the assistant's reply, with
`session_id`, `message_id` and `usage` alongside it. Give each new message
a new `id`; reusing `m1` in the same session is rejected.

`serve` binds to loopback by default, every caller is the local operator,
and only pages served from this machine may call it from a browser
([identity and access](access.md)). Put it behind your own reverse proxy
with auth before exposing it. `--host 0.0.0.0 --port 8080` changes the binding,
`--reload` restarts on source changes (`make dev` is the same thing). The
`assistant-runtime` command loads `.env` before it starts; a host that
embeds the runtime through `create_asgi_app` or `create_runtime` owns its
own environment (see [composition](composition.md)).
If a previous assistant-runtime still holds the port (recognised by its
`/health` answer), `serve` stops it and takes over, so a new configuration
takes effect with one command; anything else on the port is left alone and
reported. `--no-replace` turns the takeover off.

## 5. Connect an application

To try the browser integration, follow the [Design Studio setup](reference-app.md#run-the-application):
clone the studio, register its assistant profile, start the runtime, then start
the studio. Ask the assistant to create a document; the app applies the requested
edits and returns the results. The studio README covers its frontend prerequisites.

For your own application, send [host context and actions](host-contract.md),
render the streamed response, then return the result of each host action.
Use [the API reference](api.md) for event and continuation shapes.

### Minimal Socket.IO client

Save the following as `client.py`. It connects to the already-running server
and prints a reply; it does not implement host actions. Run it with
`uv run --with "python-socketio[client]" python client.py` to install its client dependencies.
Streaming uses the `/assistant` namespace:

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

Add Postgres when conversations, prompt-artifact versions, runtime settings
and queued inbox messages must survive a restart. From a source checkout:

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

Install the `ag-ui` extra (`uv sync --locked --extra ag-ui` in a checkout, or
`uv tool install --force "assistant-runtime[ag-ui]"` for the installed CLI) and point an AG-UI client at
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
