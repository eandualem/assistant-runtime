# Deployments

Start the runtime independently, then start each host application. Several apps
can connect to one runtime. An app's development command should check its
configured runtime URL and fail with “Start the backend first” if it is offline.
It must never launch, replace or stop the backend, including when the app exits.

## Start once, connect from each app

Configure provider credentials and optional services in the runtime's own `.env`.
Register the profiles supplied by your apps there:

```bash
# Runtime .env: registered names come from the TOML files, not their filenames.
ASSISTANT__PROFILES='["/absolute/path/to/editor/profiles/assistant.toml","/absolute/path/to/viewer/profiles/assistant.toml"]'
# Optional voice support. Keys stay in the runtime environment.
VOICE__ENABLED=true
VOICE__DELEGATION_ENABLED=false
```

From the runtime checkout, run `make dev` (HTTP and Socket.IO on
`http://127.0.0.1:7100`). Then run each app's development command in its own
terminal. The app configures its runtime URL and sends its registered profile
name. `GET /health` checks backend availability; `GET /api/artifacts/profile`
returns `available_profiles` for checking the app's profile registration.
Authentication failures must be reported as such, not treated as permission
to replace the server. Missing optional voice configuration should affect the
voice feature rather than block text chat.

CORS controls which browser origins can contact the backend; it does not start
or stop processes. The default allows localhost, `127.0.0.1` and `[::1]` on
any port, so two local apps need no CORS change. Keep the runtime bound to
loopback for local development. For remote hosting configure trusted origins
and host authentication as described in [access](access.md); CORS is not
user authentication. App server proxies still need to forward configured
credentials and enforce their own browser access rules.

## App choices and runtime controls

Each chat request, steering message and host-tool continuation can send
`profile: "editor"` at the top level. This must be a registered profile name;
requests cannot load a file. Send it consistently on every request. Omitting
it selects `AssistantDefinition.profile`, else `ASSISTANT__PROFILE`, else
`neutral`; the selector is not persisted as a session binding. Keep separate
session IDs for each app/conversation. Profiles scope prompt artifacts, not
identity or authorization: the existing principal and session ownership rules
still apply.

Artifact endpoints accept `?profile=editor`, including discovery, reads,
proposals, updates and activation. Both the system prompt and `manage_artifacts`
use the request's selected profile. App-specific persona and actions remain in
the app's profile and host contract; the runtime needs no app-specific code.

Model and thinking choices remain in request `config`; the operator can restrict
them with `ASSISTANT__REQUEST_MODELS`, budget ceilings and
`ASSISTANT__REQUEST_SERVICE_TIER`. Provider keys, subscription authentication,
optional tools, voice enablement and resource limits remain startup configuration.
For subscription-only OpenAI models set `LLM__CODEX_ONLY=true` and configure
OAuth as documented in [configuration](configuration.md).

A voice creation request can send `instructions` (1–16,000 nonblank characters)
and `profile` for its delegated backend turns. Omitted instructions use the
startup mode-specific defaults. Conversation-only calls always retain the
runtime's no-delegation guard. Call instructions cannot enable voice, bypass
the delegation ceiling or change credentials, model or resource limits.
`GET /api/voice/status` advertises `call_instructions_supported: true`.
See [voice](voice.md) for the complete contract.

## Runtime restarts belong to the operator

Only the operator restarts the runtime to apply startup configuration. The
`serve` command replaces a prior runtime on its port by default; pass
`--no-replace` to refuse an occupied port. Apps should never invoke it.
`--reload` watches runtime source changes and is for development. Without
Postgres a restart clears process memory; use `make db-up && make db-upgrade`
(or `assistant-runtime migrate`) when sessions and artifact versions must
survive restarts.

## Memory-only mode and the session TTL

Without a reachable Postgres the runtime runs entirely in process memory:

- sessions, the message tree and pending host actions;
- artifact versions the assistant or the host wrote (`durable: false` in
  every mutation response);
- the runtime settings overlay (`PATCH /api/settings` answers
  `persisted: false`) and the inbox.

A restart clears all of it. This is fine for a first look and for
development, and it is the default; a demo that people return to needs
Postgres. A Postgres-backed session then expires
`ASSISTANT__SESSION_TTL_HOURS` (24 by default, up to 168) after its last
saved activity and expired rows are deleted at startup, so raise it for a
demo that is meant to keep yesterday's conversation. In memory-only mode
the TTL does not apply: a session stays in the cache until it is evicted
or the process stops.
