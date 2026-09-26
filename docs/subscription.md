# OpenAI through a ChatGPT/Codex subscription

For local personal use, the runtime can authenticate the same way the
Codex CLI does instead of using a usage-billed `OPENAI_API_KEY`. Set
`OAUTH__ENCRYPTION_KEY` to a Fernet key (`uv run --with cryptography python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`),
start the server, and either run the device flow
(`POST /api/oauth/openai/device-code`, then open the returned URL and
enter the code) or import an existing Codex CLI login
(`codex login`, then `POST /api/oauth/openai/codex-cli/sync`).
`GET /api/oauth/openai/status` shows the connection;
`DELETE /api/oauth/openai` cancels pending device authorization and waits for any
in-flight credential update before clearing the login. Starting another device
flow replaces the old one. While connected, every `openai:`
model goes through the subscription (set `LLM__CODEX_MODELS` to a JSON
list to narrow that); the backend decides what the plan allows. Other
OpenAI features, such as image generation, still use an API key.
To make sure OpenAI is billed to the subscription and never to an API
key, set `LLM__CODEX_ONLY=true` at startup: every `openai:` model then
needs a connected subscription, models excluded by `LLM__CODEX_MODELS`
and missing or expired OAuth are rejected before an LLM request is built,
and an `OPENAI_API_KEY` does not make openai available. Other providers
with a configured key (Anthropic, Google, OpenRouter, Cerebras) stay
usable, and `ASSISTANT__REQUEST_MODELS` limits what a request may choose.
The default is `false`, which retains API fallback when OAuth is
unavailable. This guard covers the shared
LLM service; voice audio and media generation have separate credentials and
billing. Disable unwanted tools/providers separately. The Codex transport
omits unsupported sampling, response-ID chaining, and `max_output_tokens`
parameters. Use native per-turn usage limits for runtime budget enforcement;
they are not a server-side generation-token cap.
For `openai:gpt-6-astra` on Codex, host tools use Responses and a thinking budget
of `4000` maps to `low` effort. Sampling and log-probability parameters are omitted.
Account/model eligibility is still decided by the subscription backend.

To request Codex Fast mode, set `LLM__CODEX_SERVICE_TIER=fast` at startup.
It sends `service_tier: "priority"` on Codex-authenticated LLM requests without
changing the model or reasoning effort. `default` requests standard processing;
unset preserves the existing provider default. This is separate from CLI
`/fast` and does not read or modify your Codex CLI configuration. Fast mode uses
more subscription credits; see the [official speed guide](https://learn.chatgpt.com/docs/agent-configuration/speed).
It does not enable API fallback, change Voice billing, or guarantee an end-to-end
latency multiplier. `/health` reports the **requested** mode at
`components.llm_service.codex_service_tier`;
chat results report each provider response under `usage.service_tiers`, with
separate `requested` and `actual` fields. An absent or unrecognized terminal
provider tier stays `null`; requested Fast mode alone is not proof of priority
processing.

A login imported from the Codex CLI stays owned by the CLI. Refresh tokens are
single-use, so the runtime never refreshes that login itself; doing so would
sign the Codex CLI out. When the held access token is within
`OAUTH__REFRESH_BUFFER_SECONDS` of expiry (one hour by default), the runtime
re-reads the CLI's auth file and picks up the tokens the CLI has rotated as you
use it. If the file's token has expired too, which happens when the Codex CLI
goes unused for about ten days (the current token lifetime), status reports
`connected: false`, `status: "expired"` and an error telling you to run
`codex login`. The runtime picks up the new login on its next request. The
CLI's auth file is this login's only store, so it is never saved to Postgres; a
copy stored by an earlier version is recognised at startup and handed back to the
CLI, while a stored OpenAI API key is left alone. A sync returns
`source: "codex_cli"` and `persisted: false`, with `connected: true` while the
CLI's token is current (`connected: false` and `status: "expired"` otherwise).
Re-sync after a restart, or enable `OAUTH__CODEX_AUTO_SYNC`.
A device-flow login belongs to the runtime; it works without Postgres, and when
Postgres is reachable it is encrypted at rest. A disconnect during
a database outage clears this process's credentials but cannot remove an
older persisted token; repeat the disconnect after database recovery before
restarting. `DELETE` reports this as `persisted_deleted: false`; `true` confirms
that no saved token remains (including an already absent row or no database
service). Missing startup encryption
configuration returns HTTP 503 with
`type: "OAuthNotConfiguredError"`; an invalid or missing CLI auth file
returns HTTP 400 with `type: "OAuthCodexSyncError"`.
`assistant-runtime doctor` reports the state of this path. It is not
intended for multi-user hosted services.


See [configuration](configuration.md) for all settings and [deployments](deployments.md) for sharing one runtime across applications.
