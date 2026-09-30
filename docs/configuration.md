# Configuration

By default, settings are read from the environment and a `.env` file in the working
directory. Nested settings use `__` as the separator: `DATABASE__PORT`,
`LLM__PRIMARY_MODEL`, `TOOLS__PAGE_SCOPES`. `.env.example` in the
repository shows common settings with comments. Python hosts may supply one
`AppSettings` object to the [application factories](composition.md); every
settings-based service uses that object.

## Three tiers

A request sees one resolved configuration, built from three layers. The
most specific wins.

| Tier | Where it comes from | Lifetime |
|---|---|---|
| frozen | supplied `AppSettings`, or environment and `.env` at startup | the process |
| runtime overlay | `PATCH /api/settings` | until changed; persisted in Postgres when available |
| per request | the message's `config` object | that turn |

The tunables, which exist in all three tiers: `default_model`,
`thinking_budget`, `temperature`, `max_turns`, `enable_working_memory`,
`summarization_model`, `working_memory_model`, `default_image_model`,
`default_video_model`, `subagent_model`, `subagent_thinking_budget`, `codex_service_tier`.
`GET /api/settings` shows each value and which tier it came from.

## Providers and models

At least one provider must be usable.

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY` | Provider keys; set any combination |
| | Cerebras models are sent every tool non-strict: the provider rejects a request whose tools carry mixed `strict` flags, and a host action with a numeric range is never strict-compatible |
| `LLM__PROVIDERS_JSON` | Alternative: `[{"provider":"anthropic","api_key":"..."}]` (provider and key only; other fields are rejected) |
| `LLM__PRIMARY_MODEL` | Chat model, default `anthropic:claude-opus-5` |
| `LLM__SUMMARIZATION_MODEL` | History summaries and lightweight tasks, default `anthropic:claude-haiku-4-5` |
| `LLM__PROVIDER_FALLBACK` | `false`; when `true`, a model whose provider has no credentials is replaced by a configured provider's default (see below) |
| `OAUTH__ENCRYPTION_KEY` | Fernet key; enables the ChatGPT/Codex OAuth path and the encrypted provider-key store (`PUT /api/providers/{provider}/api-key`) |
| `LLM__CODEX_MODELS` | JSON list of OpenAI model names to route through the ChatGPT/Codex subscription when connected; empty routes every `openai:` model |
| `LLM__CODEX_ONLY` | `false`; when `true`, the subscription guard: every `openai:` model must go through the ChatGPT/Codex subscription (a disconnected subscription is an error, never an `OPENAI_API_KEY` fallback), and an `OPENAI_API_KEY` does not make openai available. Other providers with a configured key stay routable; `ASSISTANT__REQUEST_MODELS` limits what a request may pick. Startup-only; voice and media have separate credentials |
| `LLM__CODEX_SERVICE_TIER` | Unset by default (omit the wire field); `fast` maps to Codex wire `service_tier: "priority"`, `default` requests standard processing. The startup fallback for the `codex_service_tier` tunable, which a request or the runtime overlay can set per turn (see `ASSISTANT__REQUEST_SERVICE_TIER`). Applies to Codex-authenticated `openai:` calls only, including auxiliaries; fast consumes more subscription credits |

The configured models are used as they are: a model whose provider has no
credentials fails its turn, and `doctor` and `chat` report the missing key.
With `LLM__PROVIDER_FALLBACK=true`, if the primary or summarization model's
provider has no credentials but another provider does, that provider's
default is used instead and a warning is logged: `openai:gpt-5.6-terra` /
`openai:gpt-5.6-luna`, `google:gemini-3.1-pro-preview` /
`google:gemini-3.8-flash`, `openrouter:x-ai/grok-4.1-fast`, or
`cerebras:gpt-oss-120b` / `cerebras:qwen-3.8-27b`. The summarization model is used for history compaction unless
`HISTORY__SUMMARIZATION_MODEL` or the runtime `summarization_model`
override is set. Working-memory extraction uses
`HISTORY__WORKING_MEMORY_MODEL` or the runtime `working_memory_model`
override when set, and otherwise the summarization model.

Model ids are `provider:name`, lowercase. Providers: `anthropic`,
`openai`, `google` (Gemini through the Google AI API), `google-cloud`
(Vertex), `openrouter`, `cerebras`. `GET /api/models` lists the catalog with each
provider's status.

The model settings of a turn come in three layers, the later one winning:
the runtime's defaults for the provider (an output limit of 8192 tokens,
Anthropic cache points, OpenRouter's `data_collection: deny`, OpenAI
response chaining), then `ASSISTANT__MODEL_SETTINGS`, then the tunables.
A tunable that is not set sends nothing, so the provider's default applies.
A `thinking_budget` becomes the provider's own setting: an effort level on
current Claude models and OpenAI reasoning models, `budget_tokens` on older
Claude models (added to `max_tokens`), a thinking budget on Google and
OpenRouter. A `temperature` is left out where the model rejects it (Claude
with thinking, models without sampling settings).

### Access (`ACCESS__*`)

| Setting | Default | Meaning |
|---|---|---|
| `mode` | `trusted_local` | `trusted_local` (every caller is the local operator), `header` (an authenticating proxy sets the principal header) or `host` (`AssistantDefinition.authenticate` decides); see [access](access.md) |
| `principal_header` | `X-Assistant-Principal` | header carrying the principal id in header mode |
| `roles_header` | `X-Assistant-Roles` | comma-separated roles in header mode; `admin` administers |
| `cors_origins` | `[]` | browser origins allowed exactly (HTTP and Socket.IO); `["*"]` allows every origin without credentials |
| `cors_origin_regex` | localhost on any port | regular expression for allowed origins; clear it when exposing the server; see [access](access.md#browser-origins) |
| `local_token` | unset | `trusted_local` only: when set, callers must present it as a bearer token or Socket.IO `auth.token` |

### Assistant (`ASSISTANT__*`)

| Setting | Default | Meaning |
|---|---|---|
| `default_model` | unset (uses `LLM__PRIMARY_MODEL`) | model override |
| `thinking_budget` | unset | thinking tokens, mapped to the provider's thinking setting; unset sends none, so the provider's default applies |
| `temperature` | unset | sampling temperature where the model accepts one; unset sends none |
| `model_settings` | `{}` | native Pydantic AI `ModelSettings` for conversation turns, as JSON, passed on unchanged: for example `{"max_tokens": 32000, "thinking": "high"}` for the output limit and reasoning effort, or a provider's own keys such as `anthropic_cache`. An output limit set here is kept as it is, so on Claude models that take `budget_tokens` it must exceed the thinking budget; without one, the default limit leaves room for it. The subscription route leaves out what its backend does not accept ([subscription](subscription.md)) |
| `subagent_model` | unset (the primary model) | model for `run_subagent`; the `subagent_model` tunable's startup default |
| `subagent_thinking_budget` | unset | thinking budget for `run_subagent`; the tunable's startup default |
| `codex_service_tier` | unset (uses `LLM__CODEX_SERVICE_TIER`) | the turn's Codex service tier; the tunable's startup default |
| `request_service_tier` | `true` | whether a request's `config` may pick `codex_service_tier`; `false` pins the configured tier (fast costs more subscription credits) |
| `request_models` | `[]` (any) | model ids a request's `config` may pick for any `*_model` tunable; a request naming another keeps the host's value. Empty allows any model: fine for development, list the allowed ones for a deployment |
| `max_turns` | `10` | agent loop iterations per request |
| `enable_working_memory` | `false` | extract working memory after each turn and add it to the end of the system prompt |
| `session_ttl_hours` | `24` | Postgres-backed sessions older than this are cleaned up; `0` keeps them until they are deleted |
| `profile` | unset (neutral) | `neutral`, `technical_operator`, or the path of a TOML profile file; `AssistantDefinition.profile` takes precedence |
| `profiles` | `[]` | Additional built-in names or TOML paths registered at startup; requests select the profile name with top-level `profile`, never a path. Duplicate names fail startup. |

`max_turns`, `thinking_budget` and `subagent_thinking_budget` are ceilings:
the runtime overlay (`PATCH /api/settings`, administration) may change
them, a request's `config` can only lower them. While a budget is unset, a
request cannot turn it on.

### Per-turn budget (`ASSISTANT__BUDGET__*`)

Native Pydantic AI usage limits for one turn; unset means no limit of
that kind. `AssistantDefinition.usage_limits` sets the same limits from
host code, and where both set one the stricter applies.

| Setting | Default | Meaning |
|---|---|---|
| `tool_calls` | unset | tool calls per turn |
| `input_tokens`, `output_tokens`, `total_tokens` | unset | tokens per turn |
| `cost_usd` | unset | reported cost per turn; only enforced when the provider prices the model |

When a limit is reached the turn ends with a `final_response` /
`error` of type `usage_limit` (not retryable); the text and completed
tool results so far are saved on the assistant message. See
[concepts](concepts.md#usage-and-budgets).

### Artifacts (`ARTIFACTS__*`)

| Setting | Default | Meaning |
|---|---|---|
| `cache_ttl_seconds` | `5` | how long prompts reuse active texts read from the store; mutations invalidate at once |
| `history_limit` | `20` | versions returned by history reads |

The default assistant profile is `AssistantDefinition.profile` when set, then
`ASSISTANT__PROFILE`, then `neutral`. Register additional profiles with
`ASSISTANT__PROFILES`; see [deployments](deployments.md).

### History (`HISTORY__*`)

| Setting | Default | Meaning |
|---|---|---|
| `compaction_enabled` | `false` | run the built-in history policy (the settings below); leave it off when the application manages context itself, for example through `AssistantDefinition.capabilities` |
| `token_budget` | `100000` | history sent to the model is compacted to fit this |
| `retain_recent` | `5` | most recent messages kept verbatim |
| `protect_recent_tool_results` | `3` | most recent tool results (counted individually) never cleared |
| `message_truncation_limit` | `1000` | characters per message in summaries |
| `summarization_model`, `working_memory_model` | unset | override the models for these tasks |
| `max_memory_entries` | `15` | key decisions kept in working memory |

### Streaming (`STREAMING__*`)

| Setting | Default | Meaning |
|---|---|---|
| `max_events_per_stream` | `10000` | safety limit |
| `stream_timeout_seconds` | `300` | one turn |
| `emit_debug_events` | `false` | `assistant:debug` events with the system prompt, history and tool selection; enable only for trusted clients: the events carry the system prompt and history of the caller's own sessions |
| `client_error_detail` | `true` | include exception and provider text in errors sent to clients; set `false` on an exposed server, clients then get the error type and trace id only (the log keeps the detail). Session and access errors describe the client's own request and keep their text either way |
| `trace_retention_hours` | `168` | debug trace rows older than this are deleted at startup (Postgres only); screenshots are no longer stored in traces |

### Tools (`TOOLS__*`)

| Setting | Default | Meaning |
|---|---|---|
| `max_tools_per_request` | `64` | warn above this many tools in one request |
| `tool_timeout_seconds` | `30` | seconds a backend tool may run per attempt; a `ToolDefinition.timeout` overrides it; the model gets a `TOOL_TIMEOUT` error. Only tools declared `idempotent` are retried once on a timeout or connection error |
| `builtin_tools` | `[]` | built-in groups to enable from `time`, `screen`, `artifacts`, `subagent`, `media`, `video`; an enabled group is registered only when it can work (`media` needs an OpenAI or Google key, `video` a Runway or Luma key; point the default media models at a provider with a key) |
| `provider_capabilities` | `null` | selected runtime business capabilities from configured providers; `null` enables all configured, `[]` disables all; unknown names fail startup. `approvals` is privileged (it types into other agents' terminals) and is registered only when named here |
| `host_tools` | `{}` | tools the host executes: `{"name": {"description": "...", "parameters": <JSON schema>}}`; names must match `^[A-Za-z0-9_-]{1,64}$` |
| `host_tools_path` | unset | a JSON file with the same shape, merged over `host_tools` |
| `page_scopes` | `{}` | `{"page name": ["tool", ...]}`: backend tools allowed while the host shows that page; unlisted pages get every tool |
| `invalidations` | `{}` | `{"tool": ["domain", ...]}`: reported as `tool_result.invalidates` so the host knows what to refresh |

JSON values are given as JSON strings in the environment:
`TOOLS__PAGE_SCOPES='{"tasks": ["create_issue", "get_time"]}'`.
Selections do not filter configured host tools, MCP servers or native
assistant extensions. Runtime business capability names are `notes`,
`library`, `peers`, `rooms`, `reminders`, `activity`, `workgroups`,
`repositories`, `approvals`, `issues`, and `messaging`; they are distinct
from native [Pydantic AI capabilities](composition.md).

### Media (`MEDIA__*`)

`default_image_model` (`openai:gpt-image-1`), `default_size`
(`1024x1024`), `default_quality` (`medium`), `cache_ttl_seconds`,
`cache_max_items`; for video
`default_video_model` (`runway:gen4-turbo`), `video_poll_interval_seconds`,
`video_timeout_seconds`, `video_max_concurrent_jobs`. Video needs the
`[video]` extra and `RUNWAYML_API_SECRET` or `LUMAAI_API_KEY`.

### Decisions (`DECISIONS__*`)

Typed decisions from [TypeSafe's Jev](decisions.md), fixed at startup; these
are not request or runtime-overlay tunables. The capability is configured when
the variable named by `api_key_env` contains a non-blank value; without it `GET /api/decisions/status`
reports `configured: false` and a call returns `503`. There is no fallback.

`api_key_env` (`TYPESAFE_API_KEY`, the name of an environment variable),
`model` (`jev-latest`), `base_url` (`https://api.typesafe.ai`; `https://`, or `http://`
only for localhost, since the key travels as a bearer header),
`timeout_seconds` (`10`, >0–120), `max_questions` (`32`, 1–256, per call),
`max_state_bytes` (`262144`, 1024–8388608, the state serialised as JSON).

### Voice (`VOICE__*`)

`VOICE__DELEGATION_ENABLED` defaults to `true`. Set it to `false` for an instance that
only permits conversation-only calls; requests cannot override this ceiling.
When enabled, callers can still opt into `mode: "conversation"` per call.
`VOICE__INSTRUCTIONS_FILE` and `VOICE__CONVERSATION_INSTRUCTIONS_FILE` read the
corresponding text from a file at startup, so a prompt checked into the host
application's repository is the single source (an unreadable or empty file
fails startup). `VOICE__CONVERSATION_INSTRUCTIONS` supplies the conversation-only persona
(default: helpful, concise conversation); it is separate from delegated-mode
`VOICE__INSTRUCTIONS`, whose default asks Live to delegate.
Call creation may supply bounded `instructions` to replace the mode-specific
startup prompt for that call, or `instructions_profile` to build them from a
registered profile's artifacts (not both), and `profile` for delegated
backend work. The
conversation guard and startup enablement, credentials and resource ceilings
remain enforced. See [conversation-only voice](voice.md#conversation-only-calls).

Optional voice provider policy, fixed at startup; these are not request
or runtime-overlay tunables. Install `[voice]` and see [the frontend contract](voice.md).

`enabled` (`false`), `provider` (`live`, or `codex` for the Codex CLI's realtime
voice on its ChatGPT login), `model` (`gpt-live-1`; live only), `voice` (`marin`
for live, `cove` for codex, checked against the
[codex voices](voice.md#codex-subscription-provider)), `api_key_env`
(`OPENAI_API_KEY`, the name of an environment variable; live only), `instructions`
(neutral concise speech and backend delegation). Voice instructions are
separate from the backend assistant's profile and artifacts unless a call
names `instructions_profile`. For the codex
provider: `codex_command` (`codex`), `codex_usage_ceiling_percent` (`97`, 1–100)
and `codex_usage_check_seconds` (`15`, 5–300). The two usage settings are
deprecated and no longer limit or stop calls; account-usage admission belongs
to the provider.

`max_sessions` (`4`, 1–100), `max_duration_seconds` (`1800`, 15–7200),
`connect_timeout_seconds` (`20`, >0–60), `close_timeout_seconds` (`10`, >0–60),
`max_transcript_chars` (`200000`, 1000–1000000), `context_chars` (`16000`,
1000–24000), `retained_calls` (`100`, 1–1000), `event_buffer_size` (`512`,
16–4096). Transcript context limits apply to delegated prompts; retained
calls and replay events are bounded process-memory caches. Database
checkpoints remain until their parent session is deleted. Transcript fragment
and event-count bounds also close excessive calls. Limits are per process;
use one owning runtime process or sticky routing for a call's full lifecycle.

### Tasks (`TASKS__*`)

`enabled` (`false`): offer the task and agent tools and accept
`POST /api/tasks` and `/api/agents`; each task or agent message runs model
turns. `max_concurrent` (`4`): tasks running at once,
the rest wait in order. `max_waiting` (`100`): queued and running tasks
beyond which a start is refused. `timeout_seconds` (`900`): a task's own
time limit. `result_max_chars` (`20000`): the stored result is cut to this.

### Events and actions (`EVENT_LOG__*`, `ACTIONS__*`)

`EVENT_LOG__MAX_PAGE` and `ACTIONS__MAX_PAGE` (`500`, up to `5000`): the
most records one listing returns. An HTTP listing's `limit` is at most
`500`, so a higher value applies only to in-process callers.

### Host state (`HOST_STATE__*`)

`max_value_bytes` (`262144`): the largest value, measured as compact UTF-8
JSON. `max_page` (`1000`): the most keys one page of a namespace listing returns.

### Heartbeat (`HEARTBEAT__*`)

`enabled` (`false`), `interval_seconds` (`300`), `startup_delay_seconds`
(`300`), `message` (`[via:heartbeat]`), `from_agent` (`heartbeat`). The
heartbeat is delivered like any message from another system, into the
most recently active session; each tick is a model call, so it is off by
default.

### Database (`DATABASE__*`)

`host` (`localhost`), `port` (`5434`), `user`, `password`, `name` (all
`assistant_runtime`), `pool_size` (`5`), `pool_overflow` (`10`), `echo`
(`false`). Postgres is optional: when unreachable at startup, sessions and
runtime settings and artifact versions stay in process memory, and
database-backed endpoints return 503. A database lost after startup is
never replaced by memory: calls that need it fail until it is back (see
[persistence](persistence.md#requiring-postgres)).

- `required` (`false`): fail startup with `DatabaseUnavailableError` when
  Postgres is unreachable, instead of keeping state in memory. Set it when
  the host chooses a database, so that a restart during an outage fails
  instead of starting in memory mode. It affects startup only: either way,
  a database lost later is `degraded` on `/health` (200) and not ready on
  `/health/ready` (503).
- `migrate_on_start` (`false`): upgrade the schema to the packaged head
  before the services start; a failed upgrade fails startup with
  `MigrationError`. With `required` too, a reachable server that lacks the
  database gets it created first (the role needs the right to create
  databases).
- An empty `password` connects without one (local `trust`
  authentication). A `host` starting with `/` is a Unix-socket directory,
  for `peer` or `trust` authentication over the socket.

### Application

`APP_NAME` (`assistant-runtime`), `DEBUG` (`false`), `LOG_LEVEL` (`INFO`),
`LOG_JSON` (`false`).

## Providers

A capability (notes, a document library, ...) is offered to the model only
when a provider for it is configured. Each provider is enabled by its own
variables:

| Variable | Capability | Provider |
|---|---|---|
| `NOTES_PATH` | notes (`manage_notes`) | a folder of markdown notes with frontmatter |
| `LIBRARY_PATHS` (`name=path,name=path`, or one path) | library (`list_documents`, `read_document`) | directories of documents, each a subdirectory with a `SKILL.md`, `README.md` or `index.md` |
| `BACKBONE_URL` (+ `BACKBONE_API_KEY`) | peers, rooms, reminders, activity, workgroups, repositories | [agent-backbone](https://github.com/eandualem/agent-backbone) |
| `BACKBONE_INFRASTRUCTURE_SESSIONS` (comma-separated) | peers | session names to leave out of the active-agent list |
| `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME` | issues (`create_issue`, `search_issues`, `get_issue_details`, `comment_on_issue`, `close_issue`) | GitHub, one repository |
| `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` | messaging (`respond_telegram`) | a Telegram bot and the chat it answers in |
| `AGENT_STATE_DIR` | approvals (`list_agent_plans`, `approve_plan`, `reject_plan`), registered only when `TOOLS__PROVIDER_CAPABILITIES` names `approvals`; also enriches peers | a directory of agent state files (the Claude Code layout, `~/.claude/state`) |

Filesystem providers accept symlinks within a configured root, including a
symlink used as the root itself. Links resolving outside the root are rejected.
The host remains responsible for root configuration and filesystem permissions;
these tools do not isolate files from other local programs that can modify them.

A note move never replaces an existing destination entry, including a dangling
symlink or one another program creates while the move runs. Where the note can
be hard-linked, a move links it at the destination and then removes the source,
retaining the original file and its metadata. For that moment the note has both
names; if the process stops there or the source cannot be removed, both remain,
and deleting either one keeps the note. Between filesystems, or where the
filesystem or system policy refuses the link, a move copies the note, preserving
content, permissions, timestamps and supported extended attributes. On macOS,
copying a file with nonzero file flags fails and retains the source: Python
cannot safely preserve those flags through an opened file descriptor.

The GitHub issue tools accept arbitrary repository labels, including an empty
list. They do not require agent-routing prefixes or a particular issue-body
template. Put workflow conventions in the assistant profile; existing `from:`
and `for:` labels continue to work when supplied. The optional `priority`
argument still adds its `blocking` or `non-blocking` label.

## Integrations

| Variable | Used by |
|---|---|
| `REPO_ORGS` | optional comma-separated allowlist for repository onboarding |
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


### Codex Fast mode and returned tier

`LLM__CODEX_SERVICE_TIER=fast` is a startup default for Codex-authenticated shared
LLM calls; `default` sends `service_tier: "default"`, and unset omits the field.
It maps to `service_tier: "priority"` using Pydantic AI's supported
`openai_service_tier` setting. It does not read or modify CLI `/fast` preferences,
change model/reasoning effort, or configure separate voice/media services.
`LLM__CODEX_ONLY=true` remains the independent guard against an OpenAI API fallback.
The `codex_service_tier` tunable (request `config`, `PATCH /api/settings`, or
`ASSISTANT__CODEX_SERVICE_TIER`) chooses the tier per turn; the startup value is
the fallback when none is set.
Unsupported tier errors propagate without retrying on a different tier/provider.
The normal SDK retry policy for transient failures remains unchanged.

See the [official Codex speed guide](https://learn.chatgpt.com/docs/agent-configuration/speed)
for subscription credit tradeoffs and the
[configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
for Fast-to-priority mapping. Provider/model eligibility and processing may vary;
measure whole-turn planning latency separately from host execution. No end-to-end
speed multiplier is guaranteed.

`/health` includes `components.llm_service.codex_service_tier`, the startup **request default**.
Chat final-response and stored assistant usage can contain:

```json
{
  "service_tiers": [
    {
      "provider_response_id": "resp_example",
      "model": "gpt-5.6-sol",
      "requested": "priority",
      "actual": "priority"
    }
  ]
}
```

Each entry describes one model response. `requested` is the actual outbound wire
setting (`null` when omitted); `actual` is the recognized tier on a provider
`response.completed` or `response.incomplete` event. Missing/unrecognized values,
an interrupted stream, or a stream without such an event leave `actual: null`.
Never infer actual priority from the requested value or a startup/created event.
Host receipt continuations retain the original usage without adding a model call.
Native model messages also retain `provider_details.codex_service_tier`; auxiliary
helpers returning only aggregated token usage may not expose tier entries.
