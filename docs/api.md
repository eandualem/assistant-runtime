# HTTP & Socket.IO API

Base URL `http://127.0.0.1:7100`. HTTP routes are under `/api`; streaming
is Socket.IO on the `/assistant` namespace. There is no authentication:
bind to localhost or put a reverse proxy with auth in front. Interactive
OpenAPI docs are served at `/docs`.

## Who is calling

Every route and the Socket.IO connection establish a principal as
configured by `ACCESS__MODE` (see [access](access.md)): the local
operator by default, the `X-Assistant-Principal` / `X-Assistant-Roles`
headers behind an authenticating proxy, or the host's callback. An
unidentified caller gets `401` (a refused connection on Socket.IO); a
session that belongs to someone else `403` (`assistant:error` of type
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
does not replay missed events, so reload the session. Cancellation and
session persistence do not provide crash recovery or exactly-once external
actions; those remain separate recovery work.

Queued steering can be consumed together. Scheduling a steering request
already delivered by another turn returns a terminal `session_error`
without starting another model run.

## The message body

Sent to `POST /api/chat` and as the payload of `assistant:message`.
Keys may be camelCase; they are normalised.

| Field | Type | Notes |
|---|---|---|
| `id` | string | client-generated message id |
| `session_id` | string | created on first use |
| `content` | string | the text; may be empty on a continuation |
| `parent_id` | string, optional | branch from this message instead of the active leaf |
| `message_type` | `standard` (default) or `steering` | see concepts |
| `attachments` | list, optional | images, documents or text for the model, or a `screenshot` for `look_at_screen`; shape in [the host contract](host-contract.md) |
| `images` | list of data URLs, optional | legacy: screenshots; a top-level `screenshot` is folded in |
| `host_context` | object, optional | what the host shows, version 1 of [the host contract](host-contract.md); invalid content is a `422` |
| `config` | object, optional | per-request overrides: `default_model`, `thinking_budget`, `temperature`, `max_turns`, `enable_working_memory`, `summarization_model`, `working_memory_model`, `default_image_model`, `default_video_model`, `subagent_model` |
| `tool_call_id`, `tool_result` | continuation only | the pending host tool's call id and its result |

## Socket.IO, namespace `/assistant`

### Client to server

| Event | Payload | Effect |
|---|---|---|
| `assistant:join_session` | `{"session_id", "host_context"?}` | join the session room and warm it (tools, prompt inputs) |
| `assistant:message` | the message body | start a turn after prior-turn cleanup; ordinary messages cancel a live turn, continuations wait |
| `assistant:cancel` | `{"session_id"}` | request cancellation and snapshot persistence for the live turn |

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
client performs the action and sends `assistant:message` with
`tool_call_id` = `pending_tool_call.call_id`, `tool_result` = whatever the
action produced (any JSON), and `content` empty. The model resumes.

## Sessions

| Route | Returns |
|---|---|
| `GET /api/sessions?limit=50&offset=0` | `[{session_id, owner_id, title, turn_number, message_count, created_at}]`; the caller's own sessions, every session for an administrator |
| `GET /api/sessions/{id}` | turn count, message count, pending tool call |
| `GET /api/sessions/{id}/messages` | the root-to-leaf path, for display |
| `GET /api/sessions/{id}/tree` | every message with its `parent_id` |
| `GET /api/sessions/{id}/traces?limit=` | debug traces (Postgres) |
| `POST /api/sessions/{id}/repair` | mark host tool calls that never got a result as failed, so the session can continue |
| `DELETE /api/sessions/{id}` | delete the session |
| `PATCH /api/sessions/{id}/owner` `{"owner_id"}` | assign the session to a principal (administration; null makes it unowned) |

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
| `POST /api/artifacts/{name}/propose` `{"content", "proposed_by"?, "expected_version"?}` | new inactive version |
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
