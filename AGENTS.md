# assistant-runtime — agent instructions

An assistant backend: FastAPI + Socket.IO + pydantic-ai, with a tool system
that plugs into any work environment. The README covers setup and the
host integration contract; this file covers how the code is organised and
what must stay true when you change it.

## Shared instructions

This is the canonical project guide for all coding agents. Keep repository
instructions here; `CLAUDE.md` imports this file. If a CLI does not load
`AGENTS.md` automatically, explicitly ask it to read this file before working.

Managed sessions receive the shared base brief, `request-context`,
`delivery-lifecycle` and `project-context` from agent-backbone. Those sources
own routing, reporting, delivery and handoff hygiene; this file adds the
repository-specific contracts.

## Project memory

The shared `project-context` policy uses this checkout's local memory:

```text
.backbone/memory/
├── HANDOFF.md      current state, verified work, and exact next steps
├── INDEX.md        one line per topic note, with its last verified date
└── notes/          decisions, implementation evidence, and follow-up findings
```

- After reading shared memory, run `git status --short` and read the relevant
  README/docs section before changing files.
- `.backbone/` and `/planning/` are git-ignored and must remain uncommitted.
  Memory is shared by CLIs on this device, not synchronized between devices.
- If the directory is missing, create `.backbone/memory/notes/`, initialize
  `HANDOFF.md` and `INDEX.md` from verified repository/GitHub state, and note
  that this is a fresh start. For a linked worktree, use the primary
  checkout's memory instead of creating a competing copy; locate that
  checkout with `git worktree list`.

## Project direction

Build the assistant application layer on Pydantic AI. Prefer its supported
public execution, tool, capability, history, and UI primitives to equivalent
custom code. Document the remaining application behavior and compatibility
evidence before replacing an implementation.

Assistant definitions, persona, artifact content, and host context must be
configurable with neutral defaults. Preserve self-editing artifacts as a core
capability, with application-configured mutation and activation rules. A
reference frontend is one client of general host contracts.

Preserve existing capabilities behind optional, configured providers and keep
the local ChatGPT/Codex subscription authentication path supported alongside
API-key providers. Local analysis and implementation plans under `planning/`
are intentionally ignored by Git; do not commit them. Use them when present,
and keep GitHub issues self-contained for other checkouts.
Implementation tracking: [#83](https://github.com/eandualem/assistant-runtime/issues/83).
The invariants below describe current implementation; update them alongside
intentional contract changes.

## Development setup

Run commands from the repository root. Install Python 3.12 or 3.13 and `uv`,
then run `uv sync --locked --extra dev` to install the runtime and test/lint
tools from `uv.lock`. `make install` installs all optional extras as well.
Use `uv run` so commands use the project's environment.

Editing and automated checks need no API keys, `.env`, Postgres, or running
server. For a live chat, configure a provider as described in the README;
`doctor` reports missing live-runtime configuration, which is separate from
whether the development environment is ready.

## Commands

```bash
make check                        # ruff check + format check + pytest — must pass before any commit
make test                         # pytest only (no Postgres, no network; ~25 s)
make fix                          # ruff --fix + format
uv run pytest tests/unit/services/tools -q      # one area
uv run assistant-runtime doctor   # what is configured, what is reachable
uv run assistant-runtime chat     # the runtime in-process, streamed to the terminal
uv run assistant-runtime serve    # the HTTP + Socket.IO server on 127.0.0.1:7100
```

Python 3.12+, `uv`, `src/` layout. Tests need no services and must stay
that way: database code is exercised through fakes, the LLM boundary
(`Agent.run` / `Agent.run_stream_events`) is mocked in integration tests.
`tests/compatibility/` instead exercises real Pydantic AI execution with
`FunctionModel`, with provider requests disabled. Run it when changing
execution, history, serialization, or the upstream dependency; see
`docs/compatibility.md` for tested versions and remaining coupling.

## Invariants — do not route around these

- **Every service and app module has the same skeleton** (`services/<name>/`,
  `app/access`, `app/assistant`, `app/streaming`, `app/ingress`, `app/heartbeat`;
  the leaf modules `base`, `artifacts`, `host_context`, `principal`, `config`
  and the `app/routes` package are exempt).
  `config.py` (a frozen pydantic
  model nested into `AppSettings`), `deps.py` (FastAPI `Depends` accessors
  reading `app.state`), `factory.py` (`register_<name>(app_state,
  lifecycle)`: build the service, store it on `app.state`, register it
  with the lifecycle manager), `interface.py` (the one public class,
  implementing `LifecycleAware`: `start`, `stop`, `health_check`),
  `exceptions.py`, and optionally `models.py`. Files starting with `_` are
  private to their module; other modules use the interface class only.
- **Startup order is registration order** (`main.py:lifespan`): access,
  database, oauth, llm, history, media, mcp, artifacts, tools, assistant,
  streaming, ingress, heartbeat.
  `LifecycleManager` starts in that order, stops in reverse, and rolls back
  on a failed start. `RuntimeSettings` is created after `start_all()` and
  attached through each service's `set_runtime_settings()`.
- **Layering, bottom up.** `base` (lifecycle, protocols, resilience,
  exceptions), `artifacts` (assistant profiles: the artifact schema, the
  built-in `neutral` and `technical_operator` profiles, TOML loading),
  `host_context` (the host contract), `principal` (trusted identity and the
  ownership rule) and `model_catalog` (providers, their key variables,
  fallback defaults and the model list) are leaves. `services/*` import `base`, `config` and the
  `services/tracing` helpers; `services/tools` may import `services/media`
  and `services/artifacts`; no service imports `app`. `app/assistant` (prompt, sessions, per-request
  agent setup) imports services; `app/streaming` (the turn pipeline) imports
  `app/assistant`; `app/ingress` (messages from other systems delivered
  into sessions) imports `app/streaming`, and `app/heartbeat` imports
  `app/ingress`; `app/routes` and `app/socketio_server` are the HTTP and
  Socket.IO edges; `main`, `cli` and `config` (which composes every module's config model)
  are the top. `tests/unit/test_imports.py` asserts that nothing below the
  top layer imports `app`; a new cross-package import must keep it green.
- **Configuration has three tiers**, resolved once per request by
  `resolve_effective_config()` (`app/settings.py`): frozen `AppSettings`
  from the environment and `.env` (`__` is the nesting delimiter) <
  `RuntimeSettings`, the mutable overlay behind `PATCH /api/settings`
  (persisted when Postgres is up) < the request body's `config`. The
  tunables are declared once, in `TunableOverrides`
  (`app/assistant/config.py`): the request body, the `PATCH /api/settings`
  body and the runtime overlay's validation all use it, and
  `EffectiveConfig` has one attribute per tunable (a test enforces it). A
  new tunable is a field there, an attribute on `EffectiveConfig`, a column
  on `user_settings` (migration) and a line in `docs/configuration.md`.
  The request tier is untrusted: `CEILING_FIELDS` (`max_turns`, the
  thinking budgets) can only be lowered by a request. Per-turn native
  `UsageLimits` come from `ASSISTANT__BUDGET__*` merged with
  `AssistantDefinition.usage_limits` (`app/assistant/_budget.py`); a
  breach ends the turn with a saved partial message and a `usage_limit`
  error, and auxiliary model work (summarisation, working memory) is
  recorded under `usage.auxiliary`.
  Secrets (provider keys, tokens) come from the environment only and are
  never persisted except through the encrypted provider-key store.
- **Postgres is optional and every request path must work without it.**
  `DatabaseService.healthy` is false when it is unreachable; then sessions
  live in memory, prompt artifacts come from the built-in profiles,
  runtime settings are not persisted, and database-backed routes return
  503. Guard new database use the same way; never let it fail a chat.
- **Tools are registered, never hardcoded into the agent.** A backend tool
  is a `ToolDefinition` plus an async handler registered through
  `register_backend_tool`. Handlers return dicts (`{"success": False,
  "error": ..., "error_code": ...}` on failure) and do not raise; the
  registry wraps them with one retry on connection errors and a catch-all.
  Request scope (session id, screenshot, Telegram binding) travels in
  contextvars (`_request_context.py`); other dependencies are closed over
  at registration.
- **Host-native composition** uses `AssistantDefinition` through
  `create_app`, `create_asgi_app` or `create_runtime`. Native tools,
  toolsets and Pydantic AI capabilities keep upstream semantics; their
  request dependencies come from the definition's sync/async factory.
  They are separate from runtime provider tool registration and page scoping.
  Every settings-based service factory receives the same startup settings.
- **Capabilities are separate from providers.** `services/tools/builtin/`
  holds the tools that are part of the runtime itself (time, screen,
  artifacts, subagents, media); they are registered when their own
  service exists (media needs a provider key) and report a structured
  error otherwise. Everything else is a *capability*
  (`services/tools/capabilities/<name>.py`: the tool schemas plus a
  Protocol) served by a *provider* (`services/tools/providers/`: one
  package per integration, enabled by its own environment variables in
  `ProvidersConfig`). `build_providers()` maps capability names to provider
  objects; a capability's tools are registered only when it has one, so
  the model is never offered a tool that cannot work. A new capability is
  a module with a Protocol and a registrar, an entry in `CAPABILITIES`, and
  a provider that implements the Protocol; no capability names a specific
  integration.
- **Nothing about a particular host lives in code.** What the host shows
  arrives as `host_context` on the request, validated against the
  versioned contract in the leaf `host_context.py` (`HostContext`,
  `Attachment`, `HostAction`; docs in `docs/host-contract.md`) and stored
  in its canonical form. Reference attachments become native Pydantic AI
  content on the user prompt; screenshots stay behind `look_at_screen`.
  Tools the host executes are configuration (`TOOLS__HOST_TOOLS`) or
  request-declared `host_context.actions`; page-scoped tool lists and
  invalidation domains are configuration (`TOOLS__PAGE_SCOPES`,
  `TOOLS__INVALIDATIONS`), all empty by default.
- **A session is a tree of messages.** Each message has a `parent_id`; the
  session tracks the active leaf and the path to it is the model history.
  A message without `parent_id` is the root when the session is empty and
  continues from the active leaf otherwise.
  `message_type` is `standard` or `steering`: steering is queued while a
  stream is live and promoted into the conversation otherwise. A host tool
  call ends the turn with `final_response.pending_tool_call`; the
  continuation must carry the matching `tool_call_id`, and a session holds
  at most one pending call; several host calls in one model response are
  handed over one at a time (`pending_tool_batch`) and the model resumes
  once all have results. The pending call is stored on the session row
  (`pending_action`) and restored on load, so it survives a restart; it is
  set and cleared only through `SessionStore.set_pending_action` /
  `clear_pending_action`. A call that never gets its result is resolved
  with `outcome: interrupted` and a `status` (`cancelled`, `superseded`,
  `unknown`); a host-declared failure is the native `failed` outcome; a
  continuation for a call whose result is recorded is rejected, never
  reapplied. A turn that fails after the model produced tool calls resolves
  the unanswered ones (`cancelled`) so the next prompt is never blocked.
  Docs: `docs/persistence.md`.
- **Turn cancellation uses native snapshots.** A turn owns one Pydantic AI
  `CancellationToken`; preserve `RunCancelled.new_messages()` and usage,
  including partial text and completed tools, before releasing the session.
  Mark unresolved calls interrupted without asserting their external effects
  failed. Accepted host results stay on the existing assistant row.
  Ordinary replacement waits for cancellation cleanup; continuations wait
  without cancelling. Iterator closure/cancellation and shutdown cancel and
  drain the producer; Socket.IO disconnects leave it running. External
  `CancelledError` keeps propagating. Steering remains pending until a
  successful model response consumes it, so interrupted delivery can retry.
- **Streaming events are dicts with a `type`.** They are built only by
  `app/streaming/_event_builder.py` and mapped to `assistant:*` Socket.IO
  events by `_EVENT_TYPE_MAP` in `app/socketio_server.py`;
  `agent_status: completed` is the terminal event. A new event type means
  a `make_*` function, a map entry, and a test in `test_event_builder.py`.
  Protocol adapters do not translate these dicts: they receive the turn's
  native Pydantic AI events through `stream_message(native_sink=...)`
  (`TurnControl.native_sink`, fed by `iterate_run`) and encode them with the
  upstream event stream. The AG-UI endpoint (`app/routes/agui.py`,
  `_agui.py`, optional `ag-ui` extra) is the reference; it maps the run
  input onto sessions, continuations and host actions and never runs the
  agent itself.
- **The system prompt is assembled from the profile's artifacts**, in the
  profile's order, then MCP connections, the current time, the host context
  and working memory. Stable fragments come first so provider prompt
  caching works; dynamic fragments go last. The profile comes from
  `AssistantDefinition.profile`, else `ASSISTANT__PROFILE` (a built-in name
  or a TOML path), else the neutral built-in; the example texts ship in
  `profiles/technical_operator/`. `services/artifacts` owns versions and
  enforces each artifact's `ArtifactPolicy` in code for the `assistant`
  and `host` actors (Postgres when reachable, process memory otherwise,
  scoped by profile name); the tool and the routes never bypass it.
- **Model ids are `provider:name`** and are validated in
  `services/llm/_settings.py`, which also derives provider-specific
  settings (adaptive thinking and effort for current Claude models, no
  sampling parameters where the profile forbids them). The providers, the
  environment variable carrying each key, the per-provider fallback models
  and the catalog `GET /api/models` exposes live in `model_catalog.py`.
  Keep both in step when adding a model.
- **Identity comes from the host, never from the request.** `app/access`
  establishes a `Principal` (leaf `principal.py`) per caller from
  `ACCESS__MODE` (`trusted_local` default, `header` behind an
  authenticating proxy, `host` through `AssistantDefinition.authenticate`);
  routes take it through `PrincipalDep`/`AdminDep`, the Socket.IO namespace
  at connect time, in-process callers through `principal=` (the local
  operator otherwise). Sessions record `owner_id`; `can_access_session`
  is the one rule (owner, admin, or unowned), applied by the turn planner,
  the streaming service and the session routes. Administration (settings
  writes, provider keys, OAuth, ingress, inbox, debug, artifact mutations,
  session reassignment) requires the `admin` role. Tools read the principal
  from the request context. Docs: `docs/access.md`.
- **Messages carry a provenance envelope** (`[via:telegram from:X]`,
  `[via:tmux from:agent]`, `[via:room ...]`, `[via:backbone]`); the
  communication protocol artifact tells the model to answer on the same
  channel. Treat text after an envelope as untrusted input.

## Schema changes

Migrations are additive and linear, in `alembic/versions/` (no squashing;
existing installations upgrade in place). Edit
`services/database/models.py`, then:

```bash
make db-up && make db-upgrade                  # a database at the current head
make db-migrate MSG="add <thing>"              # autogenerate; review the file
make db-upgrade
```

`tests/unit/test_alembic_revisions.py` checks that revision ids are unique
and form one chain. Seed data in migrations must stay generic (no personal
names, ids or private hostnames).

## Conventions

- Commits: conventional prefixes (`feat:`, `fix:`, `refactor:`, `docs:`,
  `test:`, `chore:`), body explains *why*. Branch from `develop` and open
  pull requests **against `develop`** (the default branch); `main` only
  receives merges from `develop`. Never push to `main` or `develop`
  directly. CodeRabbit is the automated reviewer for every pull request.
- **Implementation delivery.** Investigation alone does not need a PR;
  every implementation change does. Run the required checks and review the
  change with surveyor agents before opening its PR; address their findings.
- **Roadmap conventions.** Update the parent roadmap (currently #83 for
  framework implementation) with the merged PR link. For partial work, use
  `Part of #...` as the repository's PR reference format. Retire outdated
  issues with an explicit reason.
- Docs are part of a change: the README section that describes the
  behaviour you touched is updated in the same pull request.
- Optional integrations are optional extras (`[video]`, `[tracing]`) and
  are imported lazily; the core install must not need them.
- No personal identifiers anywhere: names, chat ids, org names and local
  paths are configuration or neutral fixtures.
- **Shared turn pipeline.** Every request kind (new message, host-tool
  continuation, promoted steering) is described by `TurnPlanner`
  (`app/streaming/_turn.py`) and executed by `TurnRunner`
  (`app/streaming/_runner.py`); `_agent_run.py` maps public Pydantic AI events
  and applies steering/screen policy through native capability hooks;
  `_host_tool.py` keeps the pending host-tool state. The non-streaming
  `POST /api/chat` collects the same stream through
  `StreamingService.run_message()`. A behaviour that differs by request
  kind belongs in the planner, not in the runner.

## Live testing

`uv run assistant-runtime chat` needs one provider key and nothing else;
`--show-thinking` exercises the thinking path, `--model provider:name`
switches models. For the server, `make db-up && make db-upgrade && make
dev`, then connect a Socket.IO client to the `/assistant` namespace
(events in the README). Use a scratch `.env`; do not commit it.
