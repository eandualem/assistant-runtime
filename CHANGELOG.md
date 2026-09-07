# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
follows [semantic versioning](https://semver.org/spec/v2.0.0.html).

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
