# GPT-Live voice

The optional voice integration connects a browser microphone and speaker to
**GPT-Live 1** using OpenAI's Live API (an API key), or to the realtime voice of a
local **Codex CLI** signed in with ChatGPT (no API key; see
[Codex subscription provider](#codex-subscription-provider)). The runtime owns
session creation and a server connection for delegation, transcripts, usage and closing.
In delegated mode, GPT-Live sends work through the existing Pydantic AI turn pipeline, so the
configured assistant profile, capabilities, tools, host actions, access rules,
history and budgets still apply to backend work. Conversation-only mode keeps speech
and transcripts independent of backend execution.

This is the Live API, not the older Realtime API. It is not a model that can be
selected through `/api/models` or `config.default_model`: voice and the backend model are
configured independently. The reference frontend supplies WebRTC, playback and
avatar animation; this package supplies the backend contract.

## Enable

```bash
uv sync --locked --extra voice --extra dev  # source checkout
# Installed package: pip install 'assistant-runtime[voice]'
export VOICE__ENABLED=true
export OPENAI_API_KEY=...                  # server environment only
uv run assistant-runtime serve
```

With Postgres, apply migrations through `uv run assistant-runtime migrate`
before serving. The integration also works in memory without Postgres.
`GET /api/voice/status` reports `enabled`, `delegation_enabled`, `configured`, `model`, `active_calls`
and lifecycle `healthy`. It also reports `conversation_mode_supported: true`
and `call_instructions_supported: true`, so hosts can check the required call
features before opening the microphone or allocating a provider session.
`configured` means a key is present, not that model
access has been tested. Disabled or unconfigured creation returns `503`.

The default `live` provider requires a Live API key; it does not exchange Codex
login credentials for Live access. `VOICE__API_KEY_ENV` selects a different server
environment variable if desired. Keys are never accepted from the browser or
returned by voice endpoints. The `codex` provider below uses the Codex CLI's own
ChatGPT login instead. Either way, the local ChatGPT/Codex backend provider can
remain configured for delegated work; its authentication is unchanged.

As documented on September 13, 2026, GPT-Live costs **$0.05 per minute of session
duration**, with backend model usage billed independently. See the current
[model page](https://developers.openai.com/api/docs/models/gpt-live-1) for pricing
and access. Silence does not end a session. Default limits are four concurrent
calls and 30 minutes per call; configure these for the deployment.

## Browser connection

All endpoints use the host's normal runtime authentication and session ownership.
Use HTTPS outside localhost for microphone access. Create a dedicated runtime
session id or reuse a conversation that has no active turn or pending host action.
An active call reserves that session: ordinary chat, steering, ordinary cancel,
repair, deletion and owner reassignment are rejected with `409` until it closes.
Ownership is checked again after reservation admission, before any history is
returned, so a queued request cannot retain access after owner reassignment.
Use the voice control endpoints during a call.

1. Create an `RTCPeerConnection`, attach the microphone track and create the
   `oai-events` data channel **before** the offer. Attach incoming audio to an
   audio element, respecting browser playback permissions. Install data-channel
   handlers before negotiation so `session.started` is not missed.
2. Set the local offer and wait for ICE gathering to complete. POST the final
   `pc.localDescription.sdp` to `POST /api/voice/calls`:

   ```json
   {
     "session_id": "voice-demo",
     "sdp": "<complete WebRTC offer>",
     "host_context": {"actions": []},
     "config": {"default_model": "openai:gpt-5.4"}
   }
   ```

   `host_context` and `config` are optional and have the normal
   [host contract](host-contract.md) and per-request tunable validation.
   `config.default_model` selects the **backend** model. Omit it to use existing defaults.
   The example is illustrative; choose a backend model available to your provider.
   Optional `instructions` supplies this call's voice persona: 1–16,000 characters,
   with blank text rejected. It replaces the startup prompt for the selected mode
   without changing other calls or global settings. Omit it to use
   `VOICE__INSTRUCTIONS` for delegated calls or `VOICE__CONVERSATION_INSTRUCTIONS`
   for conversation-only calls; their startup file overrides still apply.
   The runtime always appends the conversation-only guard in that mode.

   Optional `profile` selects a registered assistant profile by name for backend
   delegations and their host-tool continuations. Names start with a lowercase
   letter and contain only lowercase letters, digits and underscores, up to 64
   characters. The operator registers available profiles at startup; requests
   cannot supply a profile file path. An unknown name returns `422` before a
   provider session is allocated. Omitting it retains the startup default.
   `profile` affects backend work; `instructions` affects spoken conversation.
   Neither field overrides startup voice enablement, the delegation ceiling,
   provider model, voice, key or call limits.
3. The `201` response contains `call_id`, `session_id`, `provider_session_id`,
   `transport: {type: "webrtc", sdp: "<answer>"}`, resolved `mode` and `events_url`.
   `events_url` includes the path prefix the runtime is mounted under, so it works
   through a host server that mounts the runtime's app.
   Apply `await pc.setRemoteDescription({type: "answer", sdp: result.transport.sdp})`.
   The HTTP operation already started the provider session: **do not send
   `session.start`**. Wait for `session.started` before any provider commands.
4. Subscribe to `events_url` for runtime state and delegated backend events.
   The direct data channel carries provider events for playback/UI observation;
   do not duplicate the server's delegation, commentary or close commands there.
5. On hangup, await `POST /api/voice/calls/{call_id}/close`, then stop the mic
   tracks, close the peer connection and close the event stream. Keep media and
   data transports alive during the close handshake when possible. Also close
   through this endpoint if browser negotiation fails after receiving a call id.

The runtime attaches its server connection before returning the answer. Its
initial `connected` status means the server side is attached; it is not proof of
browser playback. `active` may arrive if the sideband sees `session.started`, but
attached sidebands need not replay it. Use browser WebRTC/media state for the
avatar's listening/playing indicators. Audio analysis of the received media track
can drive mouth movement; the backend does not generate visemes.

See OpenAI's [WebRTC guide](https://developers.openai.com/api/docs/guides/voice-webrtc?api=live)
and [server controls](https://developers.openai.com/api/docs/guides/voice-server-controls?api=live).

## Conversation-only calls

Set `mode: "conversation"` on creation to keep Live focused on conversation.
`mode: "delegated"` retains the existing backend bridge. When mode is omitted,
`VOICE__DELEGATION_ENABLED=true` (the default) selects delegated mode; setting
this startup ceiling to `false` selects conversation mode and rejects explicit
requests for delegated mode with `409`, before allocating a provider session.
The resolved mode is fixed for the call and appears in creation responses and
snapshots. A context PATCH cannot change it. Set the conversation persona with
per-call `instructions`, or use the startup fallback
`VOICE__CONVERSATION_INSTRUCTIONS` (default: helpful, concise conversation).
Conversation calls never inherit `VOICE__INSTRUCTIONS`, which belongs to the
delegated mode and defaults to requesting delegation. Keep custom conversation
instructions free of delegation directives; the runtime adds a final policy
prohibiting delegation and unconfirmed execution claims.

Conversation calls keep the normal session lease, ownership, transcripts, usage
and close handshake. They create no backend worker and discard unexpected
provider delegation events without executing or returning backend results. The
voice tool-result endpoint returns `409`; voice cancel returns
`{"cancelled": false}` without affecting speech. Run independent action planning
through ordinary chat on a **separate** backend session. Never execute movement
from the old voice delegation path alongside that controller.

Live's public configuration has no disabled delegation type, so creation still
uses `delegation: {type: "client"}`. Conversation mode enforces the restriction
in the runtime, adds conversation-only instructions, and sets the provider's
`client.data_channel.allowed_client_events` array to exactly:

- `session.instructions.append`
- `session.thinking.append`
- `session.input_audio.mute`
- `session.input_audio.unmute`

The provider expects `client.data_channel` to be an **object**, with the array
under `allowed_client_events`, as documented in the
[Live API reference](https://developers.openai.com/api/reference/typescript/resources/live).
`allowed_server_events` is omitted to retain all server acknowledgments and
transcript events. Restrictions do not affect the trusted sideband.

This excludes browser `session.update` and all `response.*` commands. Hosts must
keep the same narrow policy in their client. The runtime owns close/finalization
through the sideband. Quiet application execution facts can use
`session.thinking.append` with `delegation_id: null`; these are application
context, not user speech or proof that the model has consumed the update. Keep
internal action receipts out of visible conversation and only report movement
started when the execution engine confirms it.

Creation optionally accepts a replacement text seed:

```json
{
  "session_id": "voice-conversation",
  "sdp": "<complete WebRTC offer>",
  "mode": "conversation",
  "history": [
    {"role": "user", "content": "Earlier question"},
    {"role": "assistant", "content": "Earlier spoken answer"}
  ]
}
```

`history` accepts only user/assistant text, at most 128 entries and 7,000 UTF-8
bytes of content in total. Oversized seeds and other roles return `422`; entries
remain ordered and are supplied as conversation messages, never instructions.
An explicit list replaces the backend history seed, including an empty list;
omitting it retains the existing bounded active-branch history behavior. It does
not import messages into the leased backend session. The application remains
responsible for durable conversation context and selecting relevant prior turns.

## Events, reconnect and host actions

`GET /api/voice/calls/{call_id}` returns the current snapshot: identity, model,
voice, resolved `mode`, status/reason, transcript fragments, delegation states,
`active_delegation`, `pending_tool_call`, cumulative `usage`, `finalized`, `cursor`
and `last_activity_at` (the time of the latest transcript activity, for an
application-side idle close).

`GET /api/voice/calls/{call_id}/events?after=123` streams SSE. Each event has an
integer `id` and JSON `data` with this envelope:

```json
{"type":"voice","call_id":"...","event":"transcript","data":{"role":"user","delta":"hello","start_ms":0,"end_ms":250}}
```

Persist the SSE id in the client and explicitly supply `after` when reconnecting;
`Last-Event-ID` is not consumed. A cursor ahead of the call returns `409`. Reconnect
only replays observation events and never re-executes work. The replay buffer is
bounded; a gap produces a `snapshot` event instead. Reconcile from that snapshot
rather than expecting every earlier delta. A restored database checkpoint also
produces one snapshot and closes the stream. SSE disconnects leave the call running.
For header authentication, use streaming `fetch` or an SSE client that supports
headers; native `EventSource` cannot set arbitrary authentication headers.

The envelope's `event` values are:

- `status`: `connected`, possibly `active`, `closing`, then `closed` or `interrupted`.
- `transcript`: ordered provider fragments with `role`, `delta`, `start_ms`, `end_ms`.
  Speakers can overlap. Fragments are not authoritative complete turns and an
  output transcript is not proof the user heard the audio.
- `transcript_done`: `{role, text}`, the final text of one spoken part, where the
  provider reports it (the `codex` provider does).
- `backend`: `{delegation_id, event}` where the inner event is the normal
  [runtime stream event](api.md#socketio-namespace-assistant). Render tool progress and full backend
  responses from these events or saved session history.
- `delegation`: `{id, status, pending_tool_call?}`; states include `waiting` (backend cancellation is draining), `running`,
  `pending_host`, `superseded`, `cancelled`, `failed`, `result_sent`,
  `result_accepted`, `result_rejected`. An accepted commentary command does not
  mean its result was spoken. Errors may subsequently be followed by a generic
  failure commentary, so retain the normal backend error event too. A delegation
  that arrives while the call's work queue is full is `failed` without a backend
  turn, and gets the same commentary.
- `usage`: `{usage, finalized}`. Usage is a cumulative provider snapshot; replace
  the previous value rather than summing updates.
- `provider_error`: sanitized `code` and `client_event_id`, when supplied.
- `context_updated` and `snapshot`.

These voice events are delivered through the call's SSE endpoint; they are not
also broadcast into a Socket.IO room. The normal Socket.IO event map recognizes
the `voice` envelope for host adapters that forward it.

When a delegation reaches `pending_host`, execute only the pending host action
through the frontend's existing action dispatcher. Its shape is
`{tool_name, call_id, arguments, queued?}`. Return the result to:

```text
POST /api/voice/calls/{call_id}/delegations/{delegation_id}/tool-result
```

```json
{"tool_call_id":"<pending call_id>","tool_result":{"selected":true},"tool_outcome":"success"}
```

Use `tool_outcome: "failed"` for a host-declared failure. The `202` response is
`{"accepted":true}` after the native continuation has run and recorded the result;
it can take as long as a backend turn. Duplicates, mismatches, superseded actions
and late results return `409`. If the HTTP response is lost, inspect the call and
saved session before retrying or executing the action again. A submitted result
is drained even if a newer delegation or hangup supersedes its speech response.
Several host actions from one model response are handed over one at a time.

`PATCH /api/voice/calls/{call_id}/context` with `{"host_context": {...}}` changes
structured context for subsequent backend turns, including action continuations.
It does not rewrite Live instructions or send attachments directly to the voice
model. On the `codex` provider the same request can carry a `fact` for the voice
itself (see [facts](#facts-for-the-voice)). `POST .../cancel` cancels backend work and resolves unanswered host actions
as interrupted/cancelled; it leaves the audio conversation connected. It returns
`{"cancelled": true}` only when an unfinished delegation was cancelled; a
delegation whose result is already sent, or being sent, is left as it is. Ordinary
speech interruption alone does not cancel backend work. New client delegations
supersede older unfinished delegations after native cleanup, except that submitted
host results finish being recorded first.

## Context, history and shutdown boundaries

Live client-delegation events identify work but contain no task text. The bridge
builds the backend request from recent role-labelled transcript fragments and the
host context, using the existing assistant's policies and tools. Transcript data
remains untrusted user context. The initial Live session receives bounded visible
text history from the active backend branch; tool schemas, tool results and
screenshots are not copied directly into Live. Configure voice persona separately
with `VOICE__INSTRUCTIONS`; backend artifact changes still affect backend turns.

Backend messages, tool results, native cancellation snapshots and token usage
remain in normal session history. Spoken input/output fragments, delegation state
and duration usage are checkpointed separately in `voice_calls` when Postgres is
healthy. Raw audio is never stored by the runtime. The provider session requests
`store: false`; consult OpenAI's applicable retention terms for provider processing.

For the provider's small commentary limit, the bridge returns at most 400 UTF-8
bytes of backend text plus a short pointer to the full chat result. Full results
stay in session history. This deliberately bounds commentary without an extra
summarization model call; hosts should show the full result alongside speech.

Graceful close sends `session.close` and waits for final usage, bounded by
`VOICE__CLOSE_TIMEOUT_SECONDS`. A missing final event is `interrupted` with
`finalized: false`; it does not assert that provider billing stopped. A provider
allocation or attach failure is not retried automatically, because an allocated
session may already exist. Inspect provider usage if allocation/finalization is
unconfirmed. A runtime restart does not resume the provider connection or replay
actions: a previously active checkpoint is exposed as interrupted, and a new
voice call must be created. Without Postgres, retained calls are process memory
only. Session deletion cascades persisted voice records; current session access
is checked even for cached calls.

## Codex subscription provider

`VOICE__PROVIDER=codex` routes calls through the realtime voice of the local
[Codex CLI](https://developers.openai.com/codex/cli) on its ChatGPT login. Install
the CLI and run `codex login`; no API key and no `[voice]` extra are needed.

```bash
export VOICE__ENABLED=true
export VOICE__PROVIDER=codex
uv run assistant-runtime serve
```

The runtime starts `codex app-server` itself (`VOICE__CODEX_COMMAND` names the
executable) and never reads or holds the login: the CLI authenticates. API-key
variables are removed from the CLI's environment and the login is checked for
every call (anything but a ChatGPT login is refused), so this provider never falls
back to API billing. Each call gets
an ephemeral, read-only Codex thread and a realtime v3 session; the browser's WebRTC
offer and the provider's answer pass through unchanged, and audio flows directly
between the browser and the provider. The browser contract above is the same.

**Delegation.** In delegated mode, the realtime model's delegation carries the
request text; the runtime runs it through the normal turn pipeline (profile, tools,
host actions, history) and speaks the result back into the call. The Codex CLI
also hands every delegation to the Codex agent behind the thread; the runtime
interrupts each turn that agent starts and instructs it not to act. The thread is
read-only with an empty working directory, but an interrupt can land after the
agent has begun a read-only action, so treat it as a Codex session on your
account. `delegation` status `result_accepted` means the provider queued the
spoken result, not that it was heard. Host actions, continuations and cancellation
behave as described above.

**What differs from GPT-Live.**

- The browser's data channel speaks the Codex realtime protocol. It rejects
  GPT-Live client events such as `session.thinking.append`, and an unknown client
  event ends the session. Conversation-only calls are enforced by the runtime and
  the instructions; the provider has no client-event allowlist.
- The provider reports no usage for the realtime session. `usage` holds the
  measured `duration_seconds` once the call closes. After the usual wait for the
  provider's close, the runtime waits at most `VOICE__CLOSE_TIMEOUT_SECONDS` each for
  the CLI to stop the session and to release its thread.
- Voices: `juniper`, `maple`, `spruce`, `ember`, `vale`, `breeze`, `arbor`, `sol`
  and `cove` (the default). `VOICE__VOICE` is checked against this list at startup,
  and `GET /api/voice/status` returns it as `voices`.
- `provider_error` codes are `codex_realtime_error`, `usage_limit_reached` and
  `codex_version_mismatch`, or a usage-guard reason; provider error text is logged,
  never returned.

### Facts for the voice

An application that works alongside a call (a background watcher, a controller
carrying out confirmed actions) can tell the voice what actually happened, so it
can mention it and never claims an action without a fact:

```json
{"fact": "Sent the confirmed message to the builder agent."}
```

`fact` is one plain-text statement of up to 400 characters; `host_context` may be
sent in the same request or omitted. The fact is added to the call as quiet context:
the call's instructions tell the voice to use facts when relevant and not to read
them out as they arrive. Set `"speak": true` to have the voice say it now instead.
The response `{"updated": true, "fact": {"accepted": true, "speak": false}}` means
the Codex CLI accepted the fact for the live session; a refusal returns `502`, and
a call that is closing returns `409`. Send each fact once, at most one per second
per call (`429` otherwise). On realtime v3 the CLI sends facts without a quiet
channel, so staying silent is the voice's instruction, not a protocol guarantee.
GPT-Live calls return `409` here; send facts on that provider's data channel.

**Usage window.** Realtime voice draws on the same Codex usage allowance as every
other Codex use on the machine. Before creating a call, and every
`VOICE__CODEX_USAGE_CHECK_SECONDS` during it, the runtime reads the account's
usage and refuses or stops the call when one of these holds. Each has a
machine-readable `reason`:

| `reason` | Meaning |
|---|---|
| `credits_available` | purchased credits could be charged |
| `spend_control_reached` | the account's spend control is reached |
| `usage_not_allowed` | the backend does not allow included usage now |
| `usage_limit_reached` | the backend reports a reached limit |
| `usage_window_limit` | a usage window is at `VOICE__CODEX_USAGE_CEILING_PERCENT` (97) or above |
| `usage_unreadable` | usage could not be read, or its shape is not recognised |

A refused creation returns `409` with `reason` and `allocation_status: "rejected"`.
A call stopped by the guard emits `provider_error` with the reason and closes with
reason `usage_guard` (or ends `interrupted` with `connection_lost` if the Codex CLI
does not confirm the stop). `GET /api/voice/usage` returns the current windows
(`used_percent`, `window_minutes`, `resets_at`), `credits_available`,
`spend_control_reached`, `ordinary_usage_allowed`, and the guard's `allowed` and
`reason`, so an application can show them before opening a call.

**Codex CLI versions.** The realtime interface is experimental in the CLI, and the
CLI ignores start parameters it does not know. Before the first call, the runtime
checks the installed CLI's protocol schema offline (`codex app-server
generate-json-schema`, no session and no usage) for every method, notification and
parameter this provider relies on. `GET /api/voice/status` reports the result as
`codex: {version, compatible, missing}`; an incompatible CLI refuses calls with
`503` and `reason: "codex_incompatible"` (`missing` names what it lacks, or
`app-server generate-json-schema` for a CLI too old to describe its protocol), and a
CLI that cannot be run with `reason: "codex_unavailable"`. A session that starts on a realtime
version other than v3 is closed with `codex_version_mismatch`. This provider was
developed against Codex CLI 0.157.1.

## Validation boundary

Offline tests exercise the documented HTTP shape, synthetic provider events,
authentication, session reservation, transcript/usage handling, SSE replay,
shutdown and cancellation races. Compatibility tests run real Pydantic AI
`FunctionModel` execution with backend tools and host continuations. They do not
establish successful provider access, microphone negotiation, audio quality,
latency or billing. An API-key-backed browser call remains the live acceptance
step. The thin private transport exists because the installed SDK has no Live
surface; the native Pydantic AI Realtime adapter implements a different protocol.

The `codex` provider is tested offline against a scripted app-server, and its
protocol check against the installed CLI. Its session setup, audio in both
directions and transcripts were confirmed with a real ChatGPT login during
development. Delegation, facts and the usage guard's in-call stop await the same
live acceptance, and the Codex realtime interface itself is experimental.
Its protocol source is the Codex CLI (`codex app-server generate-json-schema
--experimental`) and the open-source [Codex repository](https://github.com/openai/codex).

Protocol sources: [Live overview](https://developers.openai.com/api/docs/guides/live),
[client delegation](https://developers.openai.com/api/docs/guides/live-delegation),
and [Live conversations](https://developers.openai.com/api/docs/guides/live-conversations).


## Creation failure diagnostics

`POST /api/voice/calls` preserves the existing string `detail` on errors. Service
creation errors additionally include `allocation_status`:

- `rejected`: local startup/configuration/capacity policy prevented allocation,
  or the provider returned a known request rejection: HTTP 400, 401, 402,
  403, 404, 405, 409, 413, 415, 422 or 429.
- `unknown`: other provider statuses (including 408, nonstandard 499 and 5xx),
  network failure/timeout, malformed success, or
  sideband attachment failure after creation. This does not confirm provider
  finalization, even when local `active_calls` is zero and the session lease
  has been released.

Errors outside the voice service (for example request validation or session
ownership) may omit this field; hosts must handle absence conservatively.
Neither classification triggers an automatic retry or guarantees billing state.
A provider HTTP rejection also includes numeric `provider_status_code` and, when
it matches `req_` plus 32 hexadecimal digits, `provider_request_id`. Other request
ID formats are omitted. The same bounded metadata is logged. Raw provider error
bodies/messages, prompts/history, SDP and credentials are never included in these
diagnostics. Existing errors on other voice endpoints retain their response shape.
