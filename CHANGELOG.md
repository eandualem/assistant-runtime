# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- A login imported from the Codex CLI stays owned by the CLI. Refresh tokens
  are single-use, and the runtime used to refresh the CLI's login itself,
  which signed the Codex CLI out on that machine. The runtime now re-reads the
  CLI's auth file near expiry and never refreshes that login. If the Codex CLI
  goes unused for about ten days, its access token expires. The runtime then
  reports the subscription disconnected, with an error telling you to run
  `codex login`, and reconnects on the next request after you do.
  `OAUTH__CODEX_AUTO_SYNC` no longer fails startup when the CLI's token is
  near expiry or expired.

### Upgrading from 0.3.0

A Codex CLI login is no longer stored in Postgres: sync again after a restart,
or enable `OAUTH__CODEX_AUTO_SYNC`. If an earlier version stored a Codex CLI
login in Postgres, sync once or disconnect to remove the stored copy.

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
