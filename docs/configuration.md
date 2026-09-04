# Configuration

Everything is read from the environment, plus a `.env` file in the working
directory. Nested settings use `__` as the separator: `DATABASE__PORT`,
`LLM__PRIMARY_MODEL`, `TOOLS__PAGE_SCOPES`. `.env.example` in the
repository lists every variable with a comment.

## Three tiers

A request sees one resolved configuration, built from three layers. The
most specific wins.

| Tier | Where it comes from | Lifetime |
|---|---|---|
| frozen | environment and `.env` at startup | the process |
| runtime overlay | `PATCH /api/settings` | until changed; persisted in Postgres when available |
| per request | the message's `config` object | that turn |

The tunables that exist in all three tiers: `default_model`,
`thinking_budget`, `temperature`, `max_turns`, `enable_working_memory`,
`summarization_model`, `working_memory_model`, `default_image_model`,
`default_video_model`, `subagent_model`. `subagent_thinking_budget` exists
in the first two tiers only. `GET /api/settings` shows each value and
which tier it came from.

## Providers and models

At least one provider must be usable.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY` | Provider keys; set any combination |
| `LLM__PROVIDERS_JSON` | Alternative: `[{"provider":"anthropic","api_key":"...","base_url":null,"timeout":120,"max_retries":3}]` |
| `LLM__PRIMARY_MODEL` | Chat model, default `anthropic:claude-opus-5` |
| `LLM__SUMMARIZATION_MODEL` | History summaries and lightweight tasks, default `anthropic:claude-haiku-4-5` |
| `OAUTH__ENCRYPTION_KEY` | Fernet key; enables the ChatGPT/Codex OAuth path and the encrypted provider-key store (`PUT /api/providers/{provider}/api-key`) |

If the primary or summarization model's provider has no credentials but
another provider does, that provider's default is used instead and a
warning is logged: `openai:gpt-5.6-terra` / `openai:gpt-5.6-luna`,
`google:gemini-3.1-pro-preview` / `google:gemini-3.8-flash`,
`openrouter:x-ai/grok-4.1-fast`. Set `LLM__PRIMARY_MODEL` to choose
explicitly. The summarization model is used for history compaction unless
`HISTORY__SUMMARIZATION_MODEL` or the runtime `summarization_model`
override is set. Working-memory extraction uses
`HISTORY__WORKING_MEMORY_MODEL` or the runtime `working_memory_model`
override when set, and otherwise the summarization model.

Model ids are `provider:name`, lowercase. Providers: `anthropic`,
`openai`, `google` (Gemini through the Google AI API), `google-cloud`
(Vertex), `openrouter`. `GET /api/models` lists the catalog with each
provider's status. Provider-specific settings are derived from the model:
current Claude models get adaptive thinking with an effort level mapped
from `thinking_budget`; older ones get a fixed budget; models that reject
sampling parameters are not sent a temperature.

### Assistant (`ASSISTANT__*`)

| Setting | Default | Meaning |
|---|---|---|
| `default_model` | unset (uses `LLM__PRIMARY_MODEL`) | model override |
| `thinking_budget` | `10000` | thinking tokens; unset disables thinking |
| `temperature` | `1.0` | sampling temperature where the model accepts one |
| `max_turns` | `10` | agent loop iterations per request |
| `enable_working_memory` | `true` | extract working memory after each turn |
| `session_ttl_hours` | `24` | sessions older than this are cleaned up |

### History (`HISTORY__*`)

| Setting | Default | Meaning |
|---|---|---|
| `token_budget` | `100000` | history sent to the model is compacted to fit this |
| `retain_recent` | `5` | most recent messages kept verbatim |
| `protect_recent_tool_results` | `3` | recent tool results never summarised |
| `message_truncation_limit` | `1000` | characters per message in summaries |
| `summarization_model`, `working_memory_model` | unset | override the models for these tasks |
| `working_memory_enabled` | `true` | |
| `max_memory_entries` | `15` | key decisions kept in working memory |

### Streaming (`STREAMING__*`)

| Setting | Default | Meaning |
|---|---|---|
| `debounce_seconds` | `0.05` | text delta coalescing |
| `max_events_per_stream` | `10000` | safety limit |
| `stream_timeout_seconds` | `300` | one turn |
| `emit_debug_events` | `false` | `assistant:debug` events with the system prompt, history and tool selection; enable only for a trusted client, the socket has no authentication |

### Tools (`TOOLS__*`)

| Setting | Default | Meaning |
|---|---|---|
| `max_tools_per_request` | `64` | warn above this many tools in one request |
| `tool_timeout_seconds` | `30` | |
| `host_tools` | `{}` | tools the host executes: `{"name": {"description": "...", "parameters": <JSON schema>}}`; names must match `^[A-Za-z0-9_-]{1,64}$` |
| `host_tools_path` | unset | a JSON file with the same shape, merged over `host_tools` |
| `page_scopes` | `{}` | `{"page name": ["tool", ...]}`: backend tools allowed while the host shows that page; unlisted pages get every tool |
| `invalidations` | `{}` | `{"tool": ["domain", ...]}`: reported as `tool_result.invalidates` so the host knows what to refresh |

JSON values are given as JSON strings in the environment:
`TOOLS__PAGE_SCOPES='{"tasks": ["create_issue", "get_time"]}'`.

### Media (`MEDIA__*`)

`default_image_model` (`openai:gpt-image-1`), `default_size`
(`1024x1024`), `default_quality` (`medium`), `cache_ttl_seconds`,
`cache_max_items`, `generation_timeout_seconds`; for video
`default_video_model` (`runway:gen4-turbo`), `video_poll_interval_seconds`,
`video_timeout_seconds`, `video_max_concurrent_jobs`. Video needs the
`[video]` extra and `RUNWAYML_API_SECRET` or `LUMAAI_API_KEY`.

### Heartbeat (`HEARTBEAT__*`)

`enabled` (`true`), `interval_seconds` (`300`), `startup_delay_seconds`
(`300`), `message` (`[via:heartbeat]`), `from_agent` (`heartbeat`). The
heartbeat injects a message through the inbox, so it needs Postgres.

### Database (`DATABASE__*`)

`host` (`localhost`), `port` (`5434`), `user`, `password`, `name` (all
`assistant_runtime`), `pool_size` (`5`), `pool_overflow` (`10`), `echo`
(`false`). Postgres is optional: when unreachable, sessions and runtime
settings stay in memory, default prompt artifacts are used, and
database-backed endpoints return 503.

### Application

`APP_NAME` (`assistant-runtime`), `DEBUG` (`false`), `LOG_LEVEL` (`INFO`),
`LOG_JSON` (`false`).

## Integrations

| Variable | Used by |
|---|---|
| `BACKBONE_URL`, `BACKBONE_API_KEY` | agent, plan, meeting, schedule, swarm, telemetry and repo tools |
| `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME` | GitHub issue tools |
| `REPO_ORGS` | optional comma-separated allowlist for repository onboarding |
| `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram tools |
| `ASSISTANT_OPERATOR_NAME` | the name in envelopes on messages the assistant sends to agents |
| `MCP_CONFIG_PATH` | MCP servers file; default `mcp_servers.json` in the working directory |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_USER_ID` | tracing (`[tracing]` extra) |
| `RUNWAYML_API_SECRET`, `LUMAAI_API_KEY` | video generation (`[video]` extra) |

The MCP file has the shape `{"mcpServers": {"name": {"command": "npx",
"args": [...], "env": {...}}}}`; `mcp_servers.example.json` is a starting
point. `${VAR:-}` references in it are expanded from the environment.

## Secrets

Provider keys and tokens are read from the environment only. The one
place a key is stored is the encrypted provider-key store behind
`PUT /api/providers/{provider}/api-key`, which needs `OAUTH__ENCRYPTION_KEY`
and Postgres. Never commit `.env`.
