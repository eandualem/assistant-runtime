# Concepts

Assistant Runtime is a **Python assistant backend built on Pydantic AI**.
It receives messages, runs a configured agent, and streams the result back.
It supplies conversation state, application context, and host integration
around Pydantic AI's agent primitives. The FastAPI and Socket.IO server sits
behind an application you already have: a dashboard, an IDE, a chat client,
or a terminal.

It is not a model provider (it calls Anthropic, OpenAI, Google or
OpenRouter), not an agent orchestrator (agent-backbone does that; the
runtime talks to it through tools), and not a user interface (it feeds
one).

## Session

A conversation with an id the client chooses. A session holds a **tree of
messages**: every message has a `parent_id`, the session tracks its active
leaf, and the path from the root to that leaf is what the model sees. A
message sent without `parent_id` continues from the active leaf (the first
one is the root), so a client that never branches need not retain earlier
message ids; every message still carries its own new `id`. A client
branches from an earlier message by naming it as the parent.

With Postgres the tree and the pending host action are persisted and
survive restarts; without it sessions live in memory for the life of the
process. Sessions expire after `ASSISTANT__SESSION_TTL_HOURS` (24 by
default). See [persistence and recovery](persistence.md).

A session belongs to the principal whose message created it; other
principals cannot read or continue it, administrators can (see
[identity and access](access.md)).

## Turn

One request from the client and everything the model does in response:
thinking, tool calls, tool results, text. A turn ends with
`final_response` and then `agent_status: completed`. The non-streaming
`POST /api/chat` returns the final text; the Socket.IO transport streams
every step.

A turn is bounded by `max_turns` (agent loop iterations, 10 by default)
and by the provider's own limits.

Cancelling a turn preserves its partial assistant text, completed tool
results and available usage. Calls without a result are marked interrupted;
the runtime cannot determine whether their external effects occurred.
The stream ends with a cancellation final/error/completed envelope. A
Socket.IO disconnect leaves the turn running, while closing an in-process
event iterator or shutting down the runtime cancels and drains it.
See [turn control](api.md#turn-control) for the HTTP and socket commands.

## Message types

| `message_type` | Meaning |
|---|---|
| `standard` | A user message. Cancels a running turn on the same session, waits for its persistence and cleanup, then starts the replacement. |
| `steering` | A mid-turn nudge ("focus on X"). Queued while a turn is live and delivered to the model at its next step; promoted into a normal message when no turn is running. Never cancels anything. |

A **continuation** is a `standard` message that carries `tool_call_id` and
`tool_result`: the client has executed a host tool and is handing the
result back (see host tools below). A continuation waits for the turn that
asked for it to finish, without cancelling it.

Queued steering is acknowledged when a model request successfully finishes
consuming it. Cancellation before that point leaves the instruction pending
for the next turn. A partially completed request may therefore see the
instruction again; this is a delivery policy, not a guarantee about external
side effects.

## Tools

The model can call **backend tools**: functions the runtime executes.
The full list with schemas is at `GET /api/debug/tools`. Built-in tools
are part of the runtime and present whenever their own service is
(artifacts need Postgres, media a provider key). Capabilities (what the
assistant can do) are offered only when a provider (who does it) is
configured, so the model is never given a tool that cannot work;
configuration lists the providers.

| Group | Tools | Needs |
|---|---|---|
| time, screen | `get_time`, `look_at_screen` | nothing (a host that sends screenshots, for the screen) |
| notes | `manage_notes` | `NOTES_PATH` |
| library | `list_documents`, `read_document` | `LIBRARY_PATHS` |
| artifacts | `manage_artifacts` | Postgres |
| peers, rooms, reminders, activity, workgroups, repositories | `list_agents`, `start_agent`, `send_agent_message`, `create_meeting_room`, `add_schedule_item`, `get_delivery_status`, `create_swarm`, `onboard_repo`, ... | `BACKBONE_URL` |
| approvals | `list_agent_plans`, `approve_plan`, `reject_plan` | `AGENT_STATE_DIR` |
| issues | `create_issue`, `search_issues`, `get_issue_details`, `comment_on_issue`, `close_issue` | `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME` |
| messaging | `respond_telegram` | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` |
| media | `generate_image`, `generate_video` | an image provider key; the `[video]` extra and a Runway or Luma key |
| subagent | `run_subagent` | nothing (uses the configured model) |

A tool whose call fails returns a structured error
(`{"success": false, "error": ..., "error_code": ...}`) instead of
failing the turn.

**MCP tools** come from servers listed in `mcp_servers.json` (or the file
at `MCP_CONFIG_PATH`) and are always available; the runtime starts nothing
when the file is absent.

## Host tools

Tools the **host application executes**, not the runtime: navigate to a
page, select an item, refresh a view. They are declared in configuration
(`TOOLS__HOST_TOOLS`), so the runtime ships none. When the model calls
one, the runtime emits a `tool_call` event with `category: "host"`,
ends the turn with `final_response.pending_tool_call`, and waits. The host
performs the action and sends a continuation with the matching
`tool_call_id` and a `tool_result` (and `tool_outcome: "failed"` when the
action failed); the runtime resumes the model with that result. When the
model asks for several host actions in one response, the runtime hands them
to the host one at a time, in order: each continuation ends with the next
`pending_tool_call` and no model run, and the model resumes once all have
results. A session holds at most one pending host tool call; it is stored
on the session (with the rest of its batch) and survives a restart. A call that never gets its result is recorded as
`cancelled`, `superseded` or `unknown`, and a result is never applied
twice. See [persistence and recovery](persistence.md).

## Host context

What the host is showing right now, sent with a message (or with
`assistant_join_session` to warm the session) as `host_context`. Version 1
of the contract is a small, typed structure: descriptive `host` metadata,
the current `view` (name, description, curated `data`, `state`),
`navigation`, `actions` the host will perform when called, `attachments`,
`background` summaries, `captured_at` and an `extensions` object for
anything host-specific. Only `view.name` is required within a view; a
context without a view is fine for a service host. Unknown fields are
rejected at the edge. The last context a session received is reused for
later messages that carry none. The full shape, the attachment model and
the action protocol are in [the host contract](host-contract.md).

The view name also selects tools: `TOOLS__PAGE_SCOPES` maps a view name to
the backend tools allowed while the host shows it. Views that are not
listed get every tool.

## Prompt artifacts and profiles

The system prompt starts with named texts called artifacts. Which
artifacts exist, in what order, with what default text and who may change
them is an **assistant profile**:

- `AssistantDefinition(profile=...)` in host code (`AssistantProfile`,
  `ArtifactDefinition`, `ArtifactPolicy` from `assistant_runtime.artifacts`);
- else `ASSISTANT__PROFILE`, a built-in name or the path of a TOML file
  (the shape is documented on `load_profile_file`; the profile `name`
  scopes stored artifact versions and must match `[a-z][a-z0-9_]{0,63}`,
  so `design_studio`, not `design-studio`);
- else the built-in `neutral` profile: one required `instructions`
  artifact with a short neutral default, and an autonomous `scratchpad`.

The original technical-operator assistant (`soul`, `persona`,
`communication_protocol`, `ecosystem`, `scratchpad`) ships as the
`technical_operator` example profile; set `ASSISTANT__PROFILE=technical_operator`
to keep it on an existing installation. Its `ecosystem` text is the one to
edit first: it describes your own environment.

Each artifact has a policy, enforced in code (never by the artifact's own
text): `assistant_edit` is `none`, `propose` (new versions wait for an
authorized actor) or `autonomous` (the assistant's writes go live at once);
`assistant_activate` lets the assistant approve or roll back versions;
`host_edit` covers the HTTP routes. Every artifact is versioned: a new
version is proposed, activated or rolled back through `/api/artifacts` or
the `manage_artifacts` tool, and the active version replaces the default in
the next prompt (a short cache is invalidated on every mutation). Writers
can pass `expected_version` to fail instead of overwriting a change they
have not seen; writing content identical to the active version records
nothing. Versions live in Postgres when it is reachable and in process
memory otherwise; every mutation result and `GET /api/artifacts/profile`
report `durable`. Stored versions are scoped by the profile's `name`, so
two assistants never share artifacts; the built-ins are `neutral` and
`technical_operator`, and versions stored before profiles existed belong
to `technical_operator`.

After the artifacts come the connected MCP servers, the current time, the
host context and the session's **working memory**: a small structured
summary (goal, progress, next steps, key decisions) the runtime extracts
after each turn when `enable_working_memory` is on.

## History

Long conversations are compacted before they reach the model. The history
policy is a native `ProcessHistory` capability that Pydantic AI runs before
every model request of a turn. Within `HISTORY__TOKEN_BUDGET` (estimated)
tokens the history is sent unchanged. Over budget, older tool results
become placeholders, keeping the `HISTORY__PROTECT_RECENT_TOOL_RESULTS`
most recent ones (a cleared result keeps its outcome and metadata). If
that is not enough, messages older than the `HISTORY__RETAIN_RECENT` most
recent are replaced by a summary written with the summarisation model,
with the first user message kept verbatim. The summary is cached in the
session, so a turn with many tool calls pays for it once and later turns
extend it; the cache is in memory only, and a restarted process summarises
again. When the summarisation model fails, a short deterministic summary
is used and the turn continues.

The conversation tree is never modified: only the model input is, the
turn's own messages are always passed through verbatim, and compaction does
not write working memory (that is the separate per-turn extraction).
Dangling tool calls and orphaned tool results are repaired by Pydantic AI
itself. Set `HISTORY__COMPACTION_ENABLED=false` to replace the policy with
one supplied through `AssistantDefinition.capabilities`, for example a
Harness compaction strategy.

## Usage and budgets

Every assistant message stores the usage of the turn that produced it:
`input_tokens`, `output_tokens`, `total_tokens`, `requests`,
`tool_calls`, `cost_usd` (null unless the provider reports a price — a
missing price is never read as free) and, under `auxiliary`, the model
work the runtime did on the side: `summarization` (history compaction)
and `working_memory` (the extraction after the turn). Subagent runs
count towards the parent turn's own numbers. The same shape travels on
`final_response.usage` (working memory is extracted after that event, so
it appears on the stored message only).

Limits are native `UsageLimits`: `max_turns` is the request limit;
`ASSISTANT__BUDGET__*` or `AssistantDefinition.usage_limits` add tool
call, token and cost ceilings (the stricter wins). A request's `config`
may lower `max_turns` and the thinking budgets but never raise them past
what the host set. Reaching a limit ends the turn with a saved partial
message and a terminal `usage_limit` error.

A predictable worst case for one turn, with `max_turns=3`,
`ASSISTANT__BUDGET__TOOL_CALLS=4`, `ASSISTANT__BUDGET__OUTPUT_TOKENS=2000`
and `THINKING_BUDGET=1000`: at most three model requests, four tool
executions and 2,000 output tokens (plus the thinking tokens the provider
bills), on top of the prompt and history the runtime sends — which the
history budget (`HISTORY__TOKEN_BUDGET`) keeps bounded. Summarisation and
working-memory calls are separate, smaller requests reported under
`auxiliary`; disable the latter with `enable_working_memory=false`.

## Envelopes

Messages that arrive from somewhere other than the host carry a
provenance tag at the start of the text:

| Envelope | Origin |
|---|---|
| `[via:telegram from:<name>]` | a person on Telegram |
| `[via:tmux from:<agent>]` | an agent's terminal session |
| `[via:room room:<id> from:<sender>]` | a meeting room |
| `[via:backbone]` | a system notification from agent-backbone |
| `[via:heartbeat]` | the runtime's own periodic check-in |

The communication protocol artifact tells the model to answer on the same
channel (with `respond_telegram`, `send_agent_message` or
`send_meeting_message`). Text after an envelope is untrusted input.

## Messages from other systems

Another agent, a bot or a scheduler reaches the assistant through
`POST /api/assistant/inject` (a message with a `[via:<via> from:<from>]`
envelope) or `POST /api/inbox` (a note with a severity). Both are
delivered the way a person's steering is: into the running turn of the
target session when one is live, otherwise as a turn of its own, run in
the background with its events sent to the session's Socket.IO room. The
target is the session named in the request, else the newest session bound
to the Telegram chat, else the most recently active session. When there is
no session at all the message waits in the inbox (Postgres, or memory
without it) and is drained into the next turn that starts. The model
answers where the envelope says the message came from.

The **heartbeat** (`HEARTBEAT__ENABLED`, off by default) uses the same
path to deliver `[via:heartbeat]` every `HEARTBEAT__INTERVAL_SECONDS`, a
regular chance for the assistant to check on its work; each tick is a
model call.

## Runtime settings

Three tiers, resolved per request: the frozen configuration from the
environment, a mutable overlay changed at runtime through
`PATCH /api/settings` (persisted when Postgres is up), and per-request
overrides in the message's `config`. The most specific wins. See
[configuration](configuration.md).
