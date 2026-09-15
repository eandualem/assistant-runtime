# Deployments

One runtime process serves one host application. Everything a host needs
to vary per conversation is a request-time choice, so there is never a
reason to run a second copy of the runtime for a second kind of request.
This page is the recipe: the environment file, the launch command, how to
restart, and what to expect without Postgres.

## One instance per host

A host application talks to one runtime on one port. What differs between
its conversations is sent with the request, not baked into a process:

| Per request (`config` on the message body) | Startup (environment) |
|---|---|
| `default_model`, `thinking_budget`, `temperature` | the provider keys, `LLM__PRIMARY_MODEL`, `LLM__CODEX_ONLY` |
| `codex_service_tier` (`default` or `fast`) | `LLM__CODEX_SERVICE_TIER` as the fallback; `ASSISTANT__REQUEST_SERVICE_TIER=false` pins it |
| `subagent_model`, `subagent_thinking_budget`, the summarisation and working-memory models | the same names under `ASSISTANT__` and `HISTORY__` as defaults |
| `host_context` (what the host shows, the actions it declares) | `ASSISTANT__PROFILE` (the prompt artifacts), `TOOLS__*` |
| a separate `session_id` per conversation | `VOICE__*` (one voice service in the same process) |

A deployment restricts the request tier with `ASSISTANT__REQUEST_MODELS`
(the models a request may pick) and `ASSISTANT__REQUEST_SERVICE_TIER`;
development leaves both open.

## The recipe

Keep the environment file in the host application's repository, without
secrets; the keys live in the runtime's own `.env` or the process
environment. A file for an application that uses the ChatGPT/Codex
subscription for `openai:` models, a second provider for cheaper calls,
a checked-in prompt, and voice in the same process:

```bash
# app/runtime.env — no secrets here; CEREBRAS_API_KEY, OPENAI_API_KEY and
# OAUTH__ENCRYPTION_KEY come from the runtime's .env or the environment.
LLM__CODEX_ONLY=true                 # openai: models only through the subscription
OAUTH__CODEX_AUTO_SYNC=true          # import the local Codex CLI login at startup
LLM__PRIMARY_MODEL=openai:gpt-5.6-sol
LLM__SUMMARIZATION_MODEL=openai:gpt-5.6-luna
HISTORY__WORKING_MEMORY_MODEL=openai:gpt-5.6-luna
ASSISTANT__SUBAGENT_MODEL=cerebras:qwen-3.8-27b
ASSISTANT__THINKING_BUDGET=4000
ASSISTANT__ENABLE_WORKING_MEMORY=false
ASSISTANT__PROFILE=/absolute/path/to/app/profiles/assistant.toml
TOOLS__BUILTIN_TOOLS='["time","screen"]'   # JSON values need quoting in an env file
VOICE__ENABLED=true
VOICE__DELEGATION_ENABLED=false
VOICE__CONVERSATION_INSTRUCTIONS_FILE=/absolute/path/to/app/profiles/live-instructions.md
```

Launch it from the runtime checkout (or an installed package) with the
file applied to the environment:

```bash
uv run --env-file app/runtime.env assistant-runtime serve --host 127.0.0.1 --port 7100
```

`serve` loads the runtime's own `.env` first, then the process environment
(which `--env-file` filled) wins. Check `GET /health` for the providers,
the profile and `voice_service.enabled`, and `GET /api/settings` for the
effective tunables.

## Restart in place

To apply a runtime change, restart the same process: `serve` recognises a
previous runtime on its port through `/health` and replaces it, so the
command above is also the restart command. Do not start a second copy on
another port for a new configuration and point the application at it: that
is how a host ends up with several half-configured runtimes and
environment files. `--reload` is for editing the runtime, not for serving
a demo; a file save restarts the process and, without Postgres, clears its
memory.

Something a restart must not lose belongs in Postgres (`make db-up &&
make db-upgrade`, or `assistant-runtime migrate`): sessions, artifact
versions, runtime settings and the inbox all survive a restart there.

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

## One profile per application

`ASSISTANT__PROFILE` names a built-in profile or a TOML file whose
artifacts hold the assistant's instructions ([concepts](concepts.md)).
The file's text is the default for each artifact until a version is
activated, so a prompt checked into the application's repository is the
prompt the runtime starts with; no `PATCH /api/artifacts/...` after
start is needed. Voice prompts come from files the same way
(`VOICE__INSTRUCTIONS_FILE`, `VOICE__CONVERSATION_INSTRUCTIONS_FILE`).
