# Assistant Runtime documentation

An installed runtime prints these pages with `assistant-runtime docs <page>`
(or `assistant-runtime help <page>`), so none of this needs a checkout.

| Page | What it covers |
|---|---|
| [getting-started](getting-started.md) | Install, send a first message, then connect an application |
| [reference application](reference-app.md) | Run Design Studio and check a live model or database integration |
| [concepts](concepts.md) | The vocabulary: sessions, turns, tools, host context, artifacts, envelopes |
| [subscription](subscription.md) | Local ChatGPT/Codex login, subscription routing and Fast mode |
| [voice](voice.md) | GPT-Live or Codex-subscription voice setup, browser connection, delegation and call accounting |
| [decisions](decisions.md) | Typed decisions from a decision model: enabling, the call shape, errors and measured overhead |
| [configuration](configuration.md) | Every setting, the three configuration tiers, secrets |
| [api](api.md) | HTTP endpoints, the Socket.IO streaming contract and the AG-UI endpoint |
| [host contract](host-contract.md) | The versioned `host_context`, attachments and the action protocol a host uses |
| [identity and access](access.md) | Authentication modes, session ownership, administration, CORS |
| [deployments](deployments.md) | Independent startup, multiple apps and profiles, runtime controls, memory-only mode |
| [persistence](persistence.md) | What is stored, pending host actions across restarts, action outcomes, recovery and worker topology |
| [composition](composition.md) | Native Pydantic AI extensions and request dependencies in a Python host |
| [compatibility](compatibility.md) | Tested dependency versions and migration boundaries |

Start with getting started for terminal chat, the reference application for a
working browser client, or composition to embed the runtime in Python.
The API and host contract pages describe the request, event and action shapes.
