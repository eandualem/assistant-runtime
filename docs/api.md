# HTTP & Socket.IO API

Base URL `http://127.0.0.1:7100`. HTTP routes are under `/api`; streaming
is Socket.IO on the `/assistant` namespace. There is no authentication:
bind to localhost or put a reverse proxy with auth in front. Interactive
OpenAPI docs are served at `/docs`.

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
| `images` | list of data URLs, optional | attached images; a top-level `screenshot` is folded in |
| `host_context` | object, optional | what the host shows (shape in concepts); the legacy `machine_state` name is accepted |
| `config` | object, optional | per-request overrides: `default_model`, `thinking_budget`, `temperature`, `max_turns`, `enable_working_memory`, `summarization_model`, `working_memory_model`, `default_image_model`, `default_video_model`, `subagent_model` |
| `tool_call_id`, `tool_result` | continuation only | the pending host tool's call id and its result |

## Socket.IO, namespace `/assistant`

### Client to server

| Event | Payload | Effect |
|---|---|---|
| `assistant:join_session` | `{"session_id", "host_context"?}` | join the session room and warm it (tools, prompt inputs) |
| `assistant:message` | the message body | start a turn; a live turn on the session is cancelled unless this is a continuation |
| `assistant:cancel` | `{"session_id"}` | cancel the live turn |

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
| `assistant:final_response` | `content`, `model`, `streamed`, `session_id`?, `message_id`?, `trace_id`?, `usage`?, `error`?, `pending_tool_call`? (`{tool_name, call_id, arguments}`) |
| `assistant:error` | `type` (`validation`, `session`, `cancelled`, `internal`), `message`, `terminal`, `retry_allowed` |
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
| `GET /api/sessions?limit=50&offset=0` | `[{session_id, title, turn_number, message_count, created_at}]` |
| `GET /api/sessions/{id}` | turn count, message count, pending tool call |
| `GET /api/sessions/{id}/messages` | the root-to-leaf path, for display |
| `GET /api/sessions/{id}/tree` | every message with its `parent_id` |
| `GET /api/sessions/{id}/traces?limit=` | debug traces (Postgres) |
| `POST /api/sessions/{id}/repair` | mark host tool calls that never got a result as failed, so the session can continue |
| `DELETE /api/sessions/{id}` | delete the session |

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

## Prompt artifacts (Postgres)

| Route | Purpose |
|---|---|
| `GET /api/artifacts` | active version of every artifact |
| `GET /api/artifacts/{name}` | one active artifact |
| `GET /api/artifacts/{name}/history` | all versions |
| `POST /api/artifacts/{name}/propose` `{"content", "proposed_by"?}` | new inactive version |
| `POST /api/artifacts/{name}/approve/{version}` | activate a version |
| `POST /api/artifacts/{name}/rollback/{version}` | reactivate an older version |
| `POST /api/artifacts/{name}/actions` `{"action": "propose"\|"approve"\|"rollback", ...}` | the three above behind one endpoint |
| `PATCH /api/artifacts/scratchpad` `{"content"}` | direct, auto-approved scratchpad update |
| `DELETE /api/artifacts/{name}` | delete every version |

Names are `soul`, `persona`, `communication_protocol`, `ecosystem`,
`scratchpad`.

## Messages from other systems

| Route | Purpose |
|---|---|
| `POST /api/assistant/inject` `{"from", "via", "message", "sessionId"?, "telegramChatId"?}` | deliver a message with a `[via:<via> from:<from>]` envelope into a session; `{"status": "delivered", "session_id", "delivery": "queued"|"promoted"}` or `{"status": "queued", "inbox_id"}` when no session exists |
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
