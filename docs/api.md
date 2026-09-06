# HTTP & Socket.IO API

Base URL `http://127.0.0.1:7100`. HTTP routes are under `/api`; streaming
is Socket.IO on the `/assistant` namespace. Callers are identified as
configured by `ACCESS__MODE` (see below and [access](access.md)); the
default treats every caller as the local operator, so bind to localhost.
Interactive OpenAPI docs are served at `/docs`.

## Who is calling

Every route and the Socket.IO connection establish a principal as
configured by `ACCESS__MODE` (see [access](access.md)): the local
operator by default, the `X-Assistant-Principal` / `X-Assistant-Roles`
headers behind an authenticating proxy, or the host's callback. An
unidentified caller gets `401` (a refused connection on Socket.IO); a
session that belongs to someone else (or an unowned legacy session, for a non-administrator) `403` (`assistant:error` of type
`forbidden`); administration without the `admin` role `403`.
Administration covers `PATCH /api/settings`, providers, OAuth, ingress,
the inbox, debugging, every artifact mutation and session reassignment.

## Health

`GET /health`: `{"healthy": bool, "components": {name: {...}}}`, status
200 or 503. Postgres being down does not make the runtime unhealthy.

## Chat (non-streaming)

`POST /api/chat` with a message body (below) runs the same turn pipeline
as the socket and returns `{"content", "model", "session_id",
"turn_number", "message_id", "pending_tool_call"}` when the turn ends.
`pending_tool_call` is set when the turn stopped on a host tool call; answer
it with a continuation body (`tool_call_id`, `tool_result`). A request
that does not fit the session is a 409 and is not worth retrying: unknown
session or parent, duplicate message id, a continuation whose
`tool_call_id` is not the pending one (or nothing is pending), steering
when the session has no conversation yet. A failed run is a 500. Both
carry `{"error", "type"}`. Steering messages are rejected here
(422); use the socket.

## Turn control

`POST /api/chat/{session_id}/cancel` requests cancellation and returns
`{"cancel_requested": true}` when a cancellable turn was found, or `false`
when none was found. The response acknowledges the request; finalization
may still be running. It also works during dependency/model setup and needs
no Postgres. An idle pending host action is not a live turn to cancel.

Cancellation saves the partial assistant text, completed tool results and
available usage in the session. Outstanding calls are recorded as
`outcome: "interrupted"`; their external effects may already have occurred.
Cancellation does not undo those effects. A continuation result already
accepted from the host is retained on the same assistant message.

A connected stream receives one `final_response` with `error: true`,
`error_type: "cancelled"` and its saved `message_id`, a terminal `error`,
then `status: completed`. Text already streamed is not repeated in the
final event. The original non-streaming chat request follows the existing
run-error response; read `/api/sessions/{id}/messages` for its saved work.

New ordinary messages cancel the previous turn and wait for its cleanup
before starting. Continuations wait for the prior turn to finish without
cancelling it. Socket.IO disconnects leave execution running; reconnecting
does not replay missed events, so reload the session. What survives a
restart, how a pending host action is recovered and why duplicate
continuations are rejected is in [persistence and recovery](persistence.md);
none of it provides exactly-once external actions.

Queued steering can be consumed together. Scheduling a steering request
already delivered by another turn returns a terminal `session_error`
without starting another model run.

## AG-UI (`POST /api/agui`)

The same turn pipeline behind the [AG-UI](https://docs.ag-ui.com) protocol,
for frontends built on `@ag-ui/client`, CopilotKit or any AG-UI client. It
needs the `ag-ui` extra (`pip install "assistant-runtime[ag-ui]"`); without
it the route answers `501`. The body is an AG-UI `RunAgentInput`, the
response is Server-Sent Events encoded by Pydantic AI's `AGUIEventStream`
(`RUN_STARTED`, `TEXT_MESSAGE_*`, `THINKING_*`, `TOOL_CALL_*`,
`TOOL_CALL_RESULT`, `RUN_FINISHED` or `RUN_ERROR`).

| AG-UI | Runtime |
|---|---|
| `threadId` | the session id; ownership and administration apply as for every other route |
| last message is a `user` message | a new message appended to the session's active leaf; its `id` is the message id |
| trailing `tool` messages | continuations of the session's pending host actions, in order (`toolCallId` = the `call_id`, `content` = the result, JSON when it parses; `error` set makes it `tool_outcome: "failed"`); when a response asked for several host tools, answer them all in one run and the model resumes once |
| earlier messages | ignored: the server-side tree is the conversation; the resent transcript is not replayed into the model |
| `tools` | request-declared host actions (`host_context.actions`) for this run only: every AG-UI request carries its own host context, so a tool not sent again is not available (unlike `host_context` omitted on the message body, which reuses the session's last context); the model's call ends the run with `TOOL_CALL_*` events and the client answers with a `tool` message in its next run |
| `context` | `host_context.background` (`description` → `value`) |
| `state` | the host context itself when it carries `version: 1`; otherwise `host_context.extensions.state` |
| `forwardedProps.config` | the per-request tunable overrides (same fields as `config` in the message body) |
| `image` and `document` user content (and legacy `binary`) | reference attachments for the model (data URIs or URLs); `audio` and `video` content is not mapped |
| client disconnect | cancels the turn (partial work is saved, as for any consumer that goes away) |
| a terminal runtime error (session error, provider error, usage limit) | `RUN_ERROR` with the message; a cancelled turn ends with `RUN_FINISHED` because AG-UI has no cancelled outcome |

Not supported over AG-UI: steering, branching (`parent_id`), `resume[]`
approvals, runtime-emitted `STATE_SNAPSHOT`/`STATE_DELTA`/`MESSAGES_SNAPSHOT`,
and Socket.IO-only fields. Session routes (`/api/sessions/...`) work on the
same session, so an AG-UI client can read the tree, the pending action and
usage. A minimal browser example is in `examples/agui/index.html`.

## The message body

Sent to `POST /api/chat` and as the payload of `assistant_message`.
Keys may be camelCase; they are normalised.

| Field | Type | Notes |
|---|---|---|
| `id` | string | client-generated message id |
| `session_id` | string | created on first use |
| `content` | string | the text; may be empty on a continuation |
| `parent_id` | string, optional | the message to branch from; omitted, the message continues from the session's active leaf (the first message is the root) |
| `message_type` | `standard` (default) or `steering` | see concepts |
| `attachments` | list, optional | images, documents or text for the model, or a `screenshot` for `look_at_screen`; shape in [the host contract](host-contract.md) |
| `images` | list of data URLs, optional | legacy: screenshots; a top-level `screenshot` is folded in |
| `host_context` | object, optional | what the host shows, version 1 of [the host contract](host-contract.md); invalid content is a `422` |
| `config` | object, optional | per-request overrides: `default_model`, `thinking_budget`, `temperature`, `max_turns`, `enable_working_memory`, `summarization_model`, `working_memory_model`, `default_image_model`, `default_video_model`, `subagent_model` |
| `tool_call_id`, `tool_result` | continuation only | the pending host tool's call id and its result |
| `tool_outcome` | continuation only | `success` (default) or `failed`: the host could not perform the action; `tool_result` is then the failure the model reads |

## Socket.IO, namespace `/assistant`

### Client to server

Client-to-server event names use underscores (they are dispatched to
`on_<event>` handlers); server-to-client events use the `assistant:` prefix.
An event the server has no handler for is ignored silently, so a misspelt
name produces no response at all.

| Event | Payload | Effect |
|---|---|---|
| `assistant_join_session` | `{"session_id", "host_context"?}` | join the session room and warm it (tools, prompt inputs) |
| `assistant_message` | the message body | start a turn after prior-turn cleanup; ordinary messages cancel a live turn, continuations wait |
| `assistant_cancel` | `{"session_id"}` | request cancellation and snapshot persistence for the live turn |

### Server to client

Each event is a JSON object with the fields below; all are sent only to the
requesting socket.

| Event | Fields |
|---|---|
| `assistant:status` | `status`: `started` or `completed` (terminal) |
| `assistant:thinking_delta` | `content`, plus segment metadata |
| `assistant:text_delta` | `content`, plus segment metadata |
| `assistant:tool_call` | `tool_name`, `arguments`, `call_id`, `category` (`backend` or `host`) |
| `assistant:tool_result` | `tool_name`, `output`, `call_id`, `duration_ms`?, `invalidates`? |
| `assistant:tool_error` | `tool_name`, `error`, `call_id` |
| `assistant:final_response` | `content`, `model`, `streamed`, `session_id`?, `message_id`?, `trace_id`?, `usage`?, `error`?, `error_type`? (including `cancelled`), `pending_tool_call`? (`{tool_name, call_id, arguments}`) |
| `assistant:error` | `type`, `message`, `error_type`?, `terminal`?, `retry_allowed`?; turn errors use `type: "error"` and a specific `error_type` |
| `assistant:debug` | `type` is one of `debug_request`, `debug_system_prompt`, `debug_history`, `debug_tool_selection`, `debug_agent_config`, `debug_thinking`, `debug_final_response`, `debug_usage`, `debug_error`, `debug_completed`; off by default, on with `STREAMING__EMIT_DEBUG_EVENTS=true` |

Segment metadata on deltas: `segment_id`, `segment_index`, `delta_index`,
`segment_started`, `segment_kind` (`text` or `thinking`), so a client can
rebuild the interleaving of thinking, text and tool groups.

**Host tool flow.** `tool_call` with `category: "host"`, then
`final_response` with `pending_tool_call`, then `status: completed`. The
client performs the action and sends `assistant_message` with
`tool_call_id` = `pending_tool_call.call_id`, `tool_result` = whatever the
action produced (any JSON), and `content` empty. The model resumes.

## Usage

`final_response.usage` and each stored assistant message carry
`{input_tokens, output_tokens, total_tokens, requests, tool_calls, cost_usd, auxiliary?}`;
`cost_usd` is null when the provider reports no price, and `auxiliary`
holds `summarization` and `working_memory` usage when those ran. A turn
that reaches a usage limit ends with `final_response.error_type` and a
terminal `error.error_type` of `usage_limit` (`retry_allowed: false`),
with the partial message saved. See [concepts](concepts.md#usage-and-budgets).

A continuation's `final_response.usage` is cumulative for its assistant
message: it includes the requests made before the host action. Sum the
latest `usage` of each assistant message for a session total, not every
`final_response`.

## Sessions

| Route | Returns |
|---|---|
| `GET /api/sessions?limit=50&offset=0` | `[{session_id, owner_id, title, turn_number, message_count, created_at}]`; the caller's own sessions, every session for an administrator |
| `GET /api/sessions/{id}` | turn count, message count, `pending_action` (`tool_call_id`, `tool_name`, `arguments`, `assistant_message_id`, `queued`: further call ids from the same response still to be handed over, or null): everything a host needs to perform the waiting action and continue |
| `GET /api/sessions/{id}/messages?leaf_id=` | the root-to-leaf path for display (see below); `leaf_id` selects another leaf's path, for branch switching |
| `GET /api/sessions/{id}/tree` | every message with its `parent_id` |
| `GET /api/sessions/{id}/traces?limit=` | debug traces (Postgres) |
| `POST /api/sessions/{id}/repair` | resolve the pending host action and every call without a result as `unknown`, so the session can continue |
| `DELETE /api/sessions/{id}` | delete the session |
| `PATCH /api/sessions/{id}/owner` `{"owner_id"}` | assign the session to a principal (administration; null makes it unowned) |

### Stored messages

`GET /api/sessions/{id}/messages` returns the path from the root to the
active leaf (or to `leaf_id`), merged by time with the session's delivered
and promoted steering, ordered by time. Steering is recorded per session,
not per message, so when `leaf_id` selects another branch the same steering
rows appear next to that path too. The route is `404` for a session the
runtime does not know; a session that was joined but has no messages yet
returns `[]`. Three row shapes:

| `role` | Fields |
|---|---|
| `user` | `id`, `parent_id`, `text`, `message_type` (`standard`), `timestamp` |
| `assistant` | `id`, `parent_id`, `text` (the text segments joined), `segments`, `usage` (see [usage](#usage)), `timestamp` |
| `steering` | `id`, `text`, `message_type: "steering"`, `status` (`delivered` or `promoted`), `timestamp`; no `parent_id`, steering is outside the tree |

`segments` is the assistant message in order. Each segment carries
`segment_id` (`segment_<n>`) and `segment_index`, and is one of:

- `{"kind": "thinking", "text"}` and `{"kind": "text", "text"}`;
- `{"kind": "tool_group", "tools": [...]}`, where each tool entry has `id`
  (the call id), `name`, `input` (the arguments), and, once a result is
  recorded, `output`, plus `outcome` when it is not `success` (`failed`,
  `denied`, `interrupted`) and `status` when the action did not complete
  (`failed`, `cancelled`, `superseded`, `unknown`; see
  [persistence](persistence.md#action-outcomes)). An entry without `output`
  is the pending host action. Entries do not say whether a tool is a host
  or backend tool; the streamed `tool_call` event does (`category`), and a
  host knows its own action names.

`GET /api/sessions/{id}/tree` returns every message with `id`, `parent_id`,
`role`, `message_type`, `content` and `created_at`, for drawing branches.

## Settings, models, providers

| Route | Purpose |
|---|---|
| `GET /api/settings` | every tunable with its value and source tier |
| `PATCH /api/settings` | change the runtime overlay; omitted fields unchanged, `null` clears an override |
| `GET /api/models` | the model catalog, provider status, and the defaults a request gets now (runtime overrides applied) |
| `GET /api/providers` | providers with auth status and key source |
| `PUT /api/providers/{provider}/api-key` `{"api_key"}` | store a key encrypted (needs `OAUTH__ENCRYPTION_KEY`, Postgres) and activate it |
| `DELETE /api/providers/{provider}/api-key` | remove a stored key |
| `POST /api/oauth/openai/device-code` | start ChatGPT/Codex device auth; returns `user_code`, `verification_uri`, `expires_in` |
| `POST /api/oauth/openai/codex-cli/sync` | import the local Codex CLI auth |
| `GET /api/oauth/openai/status` | connection status |
| `DELETE /api/oauth/openai` | disconnect |

## Prompt artifacts

Versions are durable with Postgres and kept in memory otherwise; every
mutation response and the profile carry `durable`. The routes act as the
profile's `host` actor: a policy denial is `403`, an unknown name `422`, a
stale `expected_version` `409`, a missing version `404`.

| Route | Purpose |
|---|---|
| `GET /api/artifacts` | active version of every artifact, in prompt order |
| `GET /api/artifacts/profile` | the profile: artifacts, roles, policies, live versions |
| `GET /api/artifacts/{name}` | the active version, or the default text (`source: "default"`) |
| `GET /api/artifacts/{name}/history` | all versions, newest first |
| `POST /api/artifacts/{name}/propose` `{"content", "expected_version"?}` | new inactive version (`201`), attributed to the calling principal |
| `PATCH /api/artifacts/{name}` `{"content", "expected_version"?}` | new version, active at once |
| `POST /api/artifacts/{name}/approve/{version}` | activate a version |
| `POST /api/artifacts/{name}/rollback/{version}` | reactivate an older version |
| `POST /api/artifacts/{name}/actions` `{"action": "propose"\|"update"\|"approve"\|"rollback", ...}` | the four above behind one endpoint |
| `DELETE /api/artifacts/{name}` | delete every version; the default applies again |

Artifact names come from the active profile; `GET /api/artifacts/profile`
lists them. The built-in `technical_operator` profile defines `soul`,
`persona`, `communication_protocol`, `ecosystem` and `scratchpad`; the
default `neutral` profile defines `instructions` and `scratchpad`.

## Messages from other systems

| Route | Purpose |
|---|---|
| `POST /api/assistant/inject` `{"from", "via", "message", "sessionId"?, "telegramChatId"?}` | deliver a message with a `[via:<via> from:<from>]` envelope into a session; `{"status": "delivered", "session_id", "delivery"}` (`delivery` is `queued` into a live turn or `promoted` to a turn of its own) or `{"status": "queued", "inbox_id"}` when no session exists |
| `POST /api/inbox` `{"from", "message", "severity"?, "context"?}` | leave a note (`context.session_id` and `context.via` are honoured); same delivery and response as above |
| `GET /api/inbox?surfaced=` | list the queued notes (Postgres) |
| `PATCH /api/inbox/{id}/surfaced` | mark a note as surfaced (Postgres) |
| `GET /api/assistant/sessions` | sessions in the shape agent-backbone expects |

## Media and debugging

| Route | Purpose |
|---|---|
| `GET /api/media/{image_id}` | a generated image from the cache |
| `GET /api/media/video/{job_id}` | video job status |
| `GET /api/debug/tools` | the complete tool registry and MCP server status |
