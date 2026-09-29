# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Voice on the ChatGPT/Codex subscription: `VOICE__PROVIDER=codex` runs calls
  through the local Codex CLI's realtime voice on its ChatGPT login, with no API
  key and no API fallback. Delegated calls hand the spoken request to the normal
  turn pipeline and speak the result; each turn the Codex agent behind the call
  starts is interrupted. A usage guard refuses or stops calls near the Codex usage limit, or
  whenever credits could be charged, with a machine-readable `reason`.
  `GET /api/voice/usage` shows the usage windows. An offline check of the
  installed CLI's protocol is reported in `/api/voice/status`.
- Facts for a Codex voice call: `PATCH /api/voice/calls/{id}/context` accepts a
  `fact` (up to 400 characters) that the voice receives as context it is
  instructed not to read aloud (best-effort: the protocol does not guarantee
  silence), or says now with `speak: true`; the response confirms the provider
  accepted it.
- Voice calls report `transcript_done` events and `last_activity_at`, and
  `events_url` includes the path prefix the runtime is mounted under.
- Durable mode: `DATABASE__REQUIRED` makes startup fail with
  `DatabaseUnavailableError` instead of falling back to memory, and
  `DATABASE__MIGRATE_ON_START` (or `migrate()` from
  `services.database.migrations`) upgrades the schema from Python. Postgres
  connects without a password or over a Unix socket, `doctor` reports the
  schema revision, and `ASSISTANT__SESSION_TTL_HOURS=0` keeps sessions until
  they are deleted. (#191)
- Artifact proposals have a lifecycle: a version is `pending` until approved
  (`active`) or `rejected`, and a replaced one becomes `superseded`.
  `GET /api/artifact-proposals` lists proposals with a diff against the active
  text, `POST /api/artifacts/{name}/reject/{version}` rejects one, and
  `assistant:artifact_proposal` / `assistant:artifact_decision` reach the
  session's Socket.IO room. With Postgres, apply migration `0028` with
  `assistant-runtime migrate`. (#192)
- With `DATABASE__REQUIRED` and `DATABASE__MIGRATE_ON_START`, a reachable server
  without the configured database gets it created and migrated.
  `DatabaseUnavailableError` and `MigrationError` carry a `cause`
  (`unreachable`, `auth_refused`, `database_missing`, `migration_failed`), and
  `schema_status().state` reports the same for `doctor`. (#194)
- Memory per subject and shared profile text: an artifact with
  `scope = "subject"` keeps one history per request `subject`, and a session
  created with a `subject` stays bound to its profile and subject. A profile can
  `include` other profiles, declare `collections` of documents the assistant
  keeps, and bound old versions with `keep_versions`. With Postgres, apply
  migration `0029` with `assistant-runtime migrate`. (#195)
- Each assistant message records the system prompt it was produced with
  (profile, subject, artifact versions);
  `GET /api/sessions/{id}/messages/{message_id}/prompt` returns it with the
  exact text, and `GET /api/artifact-prompt` previews the prompt a turn would
  start from now. With Postgres, apply migration `0030` with
  `assistant-runtime migrate`. (#197)
- `AssistantDefinition.profiles` registers extra profiles from Python,
  alongside `ASSISTANT__PROFILES`. Host artifact writes (`propose`, `PATCH`,
  `approve`, `reject`) take an optional `label`, recorded after the principal
  in `proposed_by` or `decided_by`, for example `local:owner`. (#198)
- `max_chars` bounds an artifact's or a collection's text: a longer write is
  refused (`422`, or `artifact_too_large` from the tool). With
  `DATABASE__REQUIRED`, a changed shipped default replaces the stored seed at
  startup while only seeds were ever stored and the newest is active, so it
  never overwrites a host or assistant version. (#199)
- Background tasks, with `TASKS__ENABLED`: `POST /api/tasks` and the model's
  `start_task`, `list_tasks`, `get_task` and `cancel_task` run bounded work as
  a turn in its own session while the conversation goes on, at most
  `TASKS__MAX_CONCURRENT` at once; `task_finished` is published on
  `app.state.events`. With Postgres, apply migration `0031` with
  `assistant-runtime migrate`. (#200)
- `POST /api/voice/calls` accepts `instructions_profile`, a registered profile
  whose artifacts become the call's persona; the call records the artifact
  versions and the exact text the voice received. (#201)
- Host-written events and actions: `POST /api/events` stores an event once per
  `(source, event_id)` and steers it only into its `target_session_id`, and
  `/api/actions` keeps an action's status history, text revisions, results and
  the owner's per-recipient confirmations. A message for a session a voice call
  holds now waits for its next turn instead of failing. With Postgres, apply
  migration `0032` with `assistant-runtime migrate`. (#202)
- Host state: `/api/host-state/{namespace}/{key}` keeps small versioned JSON
  values for the host, and `expected_version` refuses a write over a version
  the writer did not see (`409`). With Postgres, apply migration `0033` with
  `assistant-runtime migrate`. (#203)
- Host cards: `POST /api/sessions/{id}/messages` appends a `host` message to an
  idle session. People see its component segments; the model reads only its
  `content`, as request-side text. With Postgres, apply migration `0034` with
  `assistant-runtime migrate`. (#205)
- A host card can start a session that does not exist yet, owned by the
  caller, and bind it with `profile` and `subject` as a first message
  would. (#207)
- Persistent agents, with `TASKS__ENABLED`: `POST /api/agents` starts an agent
  for a `profile` and `subject` that keeps one session, each message
  (`POST /api/agents/{id}/messages` or the model's `message_agent`) runs as its
  next turn, and `agent_message_finished` carries the sender's
  `parent_session_id`. With Postgres, apply migration `0035` with
  `assistant-runtime migrate`. (#211)
- Each persistent agent has its own `config` (the request tunables, such as its
  model or Codex service tier), set with `POST /api/agents` and changed with
  `PATCH /api/agents/{id}`. With Postgres, apply migration `0037` with
  `assistant-runtime migrate`. (#224)
- `GET /api/models` offers `openai:gpt-6-sol` and `openai:gpt-6-luna`, and the
  whole GPT-6 family gets the OpenAI reasoning settings (reasoning effort from
  the thinking budget, a reasoning summary, response chaining) and sends no
  temperature. (#232)

### Changed

- The model request carries only what the application asked for: no
  current-time line, working memory off by default
  (`ASSISTANT__ENABLE_WORKING_MEMORY`), and host context moved out of the
  system prompt into the user message it came with, as a `<host_context>`
  block. A message sent without `host_context` carries the session's last one,
  so omitting it does not clear it. Anthropic requests add a cache point on the
  conversation. See the
  [consumer migration note](https://github.com/eandualem/assistant-runtime/issues/210#issuecomment-5873168712).
  (#212)
- Later turns replay the conversation exactly as the model saw it, including
  steering, attachments, thinking signatures and tool arguments, from Pydantic
  AI messages stored with each message. Reference attachments are stored and
  replayed, and compaction is off by default (`HISTORY__COMPACTION_ENABLED`).
  With Postgres, apply migration `0036` with `assistant-runtime migrate`; older
  rows replay as before. See the
  [consumer migration note](https://github.com/eandualem/assistant-runtime/issues/210#issuecomment-5873168712).
  (#213)
- Nothing is on until the application enables it: `TOOLS__BUILTIN_TOOLS` is
  empty, the `neutral` profile is empty, artifacts default to
  `assistant_edit: none`, thinking and temperature are unset (native settings
  go in `ASSISTANT__MODEL_SETTINGS`), provider fallback needs
  `LLM__PROVIDER_FALLBACK`, and steering reaches the model unframed. See the
  [consumer migration note](https://github.com/eandualem/assistant-runtime/issues/210#issuecomment-5873168712).
  (#214)
- The choice between Postgres and memory is made once, at startup: `/health` no
  longer switches it, and a database chosen at startup is tried again on each
  call, so the runtime recovers when it returns. During an outage,
  `PATCH /api/settings`, `POST /api/inbox` and `POST /api/assistant/inject`
  answer `503` instead of keeping data in memory, and a turn fails instead of
  building its prompt from the defaults. (#221)
- `GET /health` reports a top-level `status`: `ok`, `degraded` (200: a
  database chosen at startup does not answer) or `unhealthy` (503). The new
  `GET /health/ready` answers 200 only when `status` is `ok`, and the database
  component reports `ready`. `DATABASE__REQUIRED` now decides startup only: a
  database lost later no longer turns `/health` into a 503. Point liveness
  probes at `/health` and readiness probes at `/health/ready`. (#226)

### Fixed

- A GPT-Live session that the provider created but the runtime could not attach
  to is now hung up, best effort, as Codex sessions already were. Previously
  nothing closed it.
- A stored OpenAI API key and a ChatGPT/Codex login no longer overwrite or
  delete each other. They shared one database row: storing the key erased a
  saved login, a new login replaced the key, and removing either removed both.
  Each now has its own row. Existing Postgres installations must apply
  migration `0027` with `assistant-runtime migrate` (or `make db-upgrade`)
  before running this version; it keeps every stored key and login.
- A login imported from the Codex CLI stays owned by the CLI. Refresh tokens
  are single-use, and the runtime used to refresh the CLI's login itself,
  which signed the Codex CLI out on that machine. The runtime now re-reads the
  CLI's auth file near expiry and never refreshes that login. If the Codex CLI
  goes unused for about ten days, its access token expires. The runtime then
  reports the subscription disconnected, with an error telling you to run
  `codex login`, and reconnects on the next request after you do.
  `OAUTH__CODEX_AUTO_SYNC` no longer fails startup when the CLI's token is
  near expiry or expired.
- A provider failure keeps the turn's partial text, completed tool results and
  usage; disallowed browser Origins are refused before state-changing HTTP
  requests; notes and library operations can no longer be redirected by a
  symlink swapped in meanwhile; lightweight CLI commands start about ten times
  faster. (#161)
- `assistant-runtime chat` checks at startup that the model it will use has
  credentials and exits with setup guidance, instead of failing on the first
  message with a library error. `serve` still starts without a key. (#165)
- A default install without Postgres no longer warns at startup or logs
  `Failed to persist trace` after every turn; it logs once, at info level, that
  sessions are kept in memory. The warning stays when a `DATABASE__*` setting
  is given. (#166)
- Parallel note moves into one folder no longer replace a note, a failed
  cross-device move no longer leaves a partial file, and a local session
  command that times out or is cancelled finishes its cleanup within 5 seconds
  and reports the cancellation. (#169)
- A note created while a same-named note is being moved into its folder is no
  longer lost; it takes the next free dated name. (#170)
- A note move never replaces an existing destination, including one another
  program creates while the move runs: the note is hard-linked into place, or
  copied without replacing where the link is refused. (#186)
- Cancelling a voice call after its delegation finished no longer relabels it
  `cancelled` or cuts off its result; the cancel reports `cancelled: false`.
  (#183)
- A full voice work queue fails the new delegation, with spoken failure
  commentary, instead of ending the call or leaving a delegation `waiting`
  forever. (#184)
- A new voice delegation, or a cancel, no longer cuts off the previous
  delegation's result while it is being sent; that result is delivered
  first. (#190)
- A request's `config.subagent_model` and `config.subagent_thinking_budget` now
  reach `run_subagent`, which used only the runtime overlay and the
  defaults. (#196)
- A route that uses Postgres answers `503` (`"type": "DatabaseError"`) when the
  database is lost after startup, instead of a `500` with the driver's
  error. (#219)
- With Postgres, a message id already used in another session is refused with
  `409` on `POST /api/chat` and `POST /api/sessions/{id}/messages`, instead of
  a `500` that included the SQL statement. (#225)
- With Postgres, a steering id already used in another session is refused
  with the duplicate error, instead of sending the Socket.IO client the
  database error with its SQL statement. (#229)
- Name checks refuse a value with a trailing newline: host-state namespaces and
  keys, artifact, collection and profile names, local session names, note
  filenames, confirmation hashes and help page names. Before, `$` let
  `PUT /api/host-state/a%0A/key` store the namespace `"a\n"`. (#230)

### Upgrading from 0.3.0

A Codex CLI login is no longer stored in Postgres: sync again after a restart,
or enable `OAUTH__CODEX_AUTO_SYNC`. A copy stored by an earlier version is
recognised at startup while it still matches the CLI's login, removed, and
handed back to the CLI. If the CLI has refreshed its login since, sync once or
disconnect while the database is reachable to remove the old copy; a disconnect
reports `persisted_deleted: true` once it is gone.

Postgres installations apply migrations `0027` to `0037` with
`assistant-runtime migrate` (or `make db-upgrade`) before running this
version. Nothing is on by default any more (#212–#214): built-in tools,
working memory, compaction, thinking, temperature and provider fallback take
effect only when the application configures them, and host context moved from
the system prompt into the user message. The
[consumer migration note](https://github.com/eandualem/assistant-runtime/issues/210#issuecomment-5873168712)
lists what a host must now configure to keep a behaviour it relied on.

## 0.3.0 — 2026-09-18

- Typed decisions: `POST /api/decisions` sends program state and a set of
  `choice`, `score` and `noul` questions to TypeSafe's Jev decision model in one
  call and returns typed answers with probabilities, usage and per-call timing.
  The runtime holds `TYPESAFE_API_KEY`; without it the capability reports
  `configured: false` and a call fails with `503`, never a language-model
  fallback. See [decisions](https://github.com/eandualem/assistant-runtime/blob/main/docs/decisions.md).

### Upgrading from 0.2.0

No migration and no new extra. Set `TYPESAFE_API_KEY` on the runtime to enable
decisions; everything else is unchanged.

## 0.2.0 — 2026-09-16

- GPT-Live voice over WebRTC, with optional delegation into the shared turn
  pipeline or conversation-only calls alongside an independent controller.
- One independently started runtime can serve multiple applications using
  registered profiles, request-selected artifacts and per-call voice instructions.
- Silent host decisions return one action or a structured hold; execution receipts
  can be saved without another model call.
- Local ChatGPT/Codex subscription authentication works without Postgres, with
  subscription-only routing, requested Fast mode and observed response tiers.
- Safer localhost origin defaults, authenticated health detail and runtime controls
  for model requests, delegated work and tool execution.
- Corrected session eviction, cancellation, working-memory ordering and usage,
  admission locking, provider retry behavior and credential handling.
- Generic prompts and tool descriptions, a demos-first README, packaged setup
  guides and the CLI `--version` flag.

### Upgrading from 0.1.0

Install `assistant-runtime[voice]==0.2.0` if the application uses voice; enable
voice and configure its API key separately from backend model authentication.
Start the runtime yourself before starting applications. Register application
profile paths with `ASSISTANT__PROFILES`; clients select the registered name.
See [deployment](https://github.com/eandualem/assistant-runtime/blob/main/docs/deployments.md)
and [voice](https://github.com/eandualem/assistant-runtime/blob/main/docs/voice.md).

Postgres users must run `assistant-runtime migrate` before restarting the runtime;
the migration chain now ends at `0026` (voice calls, trace indexing, service-tier
settings and steering profile selection). Migration `0024` permanently clears
historical screenshot payloads from traces; it does not delete conversations.
Memory-only installations need no database migration.
Review [access configuration](https://github.com/eandualem/assistant-runtime/blob/main/docs/access.md)
if the application is served from a non-local origin.

## 0.1.0

First public release.

- An assistant backend on FastAPI, Socket.IO and Pydantic AI: sessions as
  message trees, streamed turns, steering, cancellation with saved partial
  work, and usage budgets.
- A registered tool system: built-in tools, capabilities served by configured
  providers, MCP servers, and host tools the calling application executes.
- A versioned host contract (`host_context`) with attachments, request
  declared actions, and host-tool continuations that survive a restart.
- Configurable assistant profiles with policy-enforced, assistant-editable
  prompt artifacts.
- An AG-UI endpoint alongside the Socket.IO transport.
- Optional Postgres: every request path works without it.
