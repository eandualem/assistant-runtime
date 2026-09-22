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

- It binds to `127.0.0.1` and defaults to trusting local callers. `ACCESS__MODE`
  defaults to `trusted_local`, which treats every caller as the local
  operator, including administration.
- CORS defaults to `localhost`, `127.0.0.1` and `[::1]` origins on any port,
  for both HTTP and Socket.IO. This limits browser access; it does not
  authenticate callers. `ACCESS__LOCAL_TOKEN` can require a bearer token
  from local clients.
- Provider keys are read from the environment or `.env` and are never written
  to the database, except through the encrypted provider-key store.

Before exposing it to anything but localhost, set `ACCESS__MODE=header` behind
an authenticating proxy or set `ACCESS__MODE=host` with your own
`authenticate` callback. Clear `ACCESS__CORS_ORIGIN_REGEX` and set
`ACCESS__CORS_ORIGINS` to the application's exact browser origins. See [docs/access.md](docs/access.md).

Tools the assistant can call reach real systems. Only enable the capability
providers you intend it to use; each one is off until its own environment
variables are set.
