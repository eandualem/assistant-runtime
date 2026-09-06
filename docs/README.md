# Assistant Runtime documentation

An installed runtime prints these pages with `assistant-runtime docs <page>`
(or `assistant-runtime help <page>`), so none of this needs a checkout.

| Page | What it covers |
|---|---|
| [concepts](concepts.md) | The vocabulary: sessions, turns, tools, host context, artifacts, envelopes |
| [getting-started](getting-started.md) | Install, one provider key, chat from the terminal, then the server and a client |
| [configuration](configuration.md) | Every setting, the three configuration tiers, secrets |
| [api](api.md) | HTTP endpoints, the Socket.IO streaming contract and the AG-UI endpoint |
| [persistence](persistence.md) | What is stored, pending host actions across restarts, action outcomes, recovery and worker topology |
| [composition](composition.md) | Native Pydantic AI extensions and request dependencies in a Python host |
| [compatibility](compatibility.md) | Tested dependency versions and migration boundaries |
