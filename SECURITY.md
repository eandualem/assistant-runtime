# Security

## Reporting a vulnerability

Report privately through
[GitHub security advisories](https://github.com/eandualem/assistant-runtime/security/advisories/new)
rather than in a public issue. Please include what you ran and what you
observed. This is a small project: expect an acknowledgement within a few
days.

## What this software assumes

The runtime is built to run next to the person using it, and its defaults say
so:

- It binds to `127.0.0.1` and ships **no authentication**. `ACCESS__MODE`
  defaults to `trusted_local`, which treats every caller as the local
  operator, including administration.
- CORS defaults to any origin, which is safe only while the server is bound
  to loopback.
- Provider keys are read from the environment or `.env` and are never written
  to the database, except through the encrypted provider-key store.

Before exposing it to anything but localhost, set `ACCESS__MODE=header` behind
an authenticating proxy or supply your own `authenticate` callback, and set
`ACCESS__CORS_ORIGINS`. See [docs/access.md](docs/access.md).

Tools the assistant can call reach real systems. Only enable the capability
providers you intend it to use; each one is off until its own environment
variables are set.
