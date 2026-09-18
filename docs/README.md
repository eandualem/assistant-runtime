# Assistant Runtime documentation

An installed runtime prints these pages with `assistant-runtime docs <page>`
(or `assistant-runtime help <page>`), so none of this needs a checkout.

| Page | What it covers |
|---|---|
| [concepts](concepts.md) | The vocabulary: sessions, turns, tools, host context, artifacts, envelopes |
| [getting-started](getting-started.md) | Install, one provider key, chat from the terminal, then the server and a client |
| [subscription](subscription.md) | Local ChatGPT/Codex login, subscription routing and Fast mode |
| [voice](voice.md) | GPT-Live setup, browser connection, delegation and call accounting |
| [decisions](decisions.md) | Typed decisions from a decision model: enabling, the call shape, errors and measured overhead |
| [configuration](configuration.md) | Every setting, the three configuration tiers, secrets |
| [api](api.md) | HTTP endpoints, the Socket.IO streaming contract and the AG-UI endpoint |
| [host contract](host-contract.md) | The versioned `host_context`, attachments and the action protocol a host uses |
| [identity and access](access.md) | Authentication modes, session ownership, administration, CORS |
| [deployments](deployments.md) | Independent startup, multiple apps and profiles, runtime controls, memory-only mode |
| [persistence](persistence.md) | What is stored, pending host actions across restarts, action outcomes, recovery and worker topology |
| [composition](composition.md) | Native Pydantic AI extensions and request dependencies in a Python host |
| [compatibility](compatibility.md) | Tested dependency versions and migration boundaries |

For the complete browser application, separately runnable live smoke checks and
current adoption limits, see [reference application](reference-app.md).
