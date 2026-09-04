# Concepts

Assistant Runtime is an **assistant backend**: it receives messages, runs a
model with a set of tools, and streams the result back. It is meant to sit
behind a host application you already have (a dashboard, an IDE, a chat
client, a terminal) rather than to be a product on its own. Built on
FastAPI, Socket.IO and pydantic-ai.

It is not a model provider (it calls Anthropic, OpenAI, Google or
OpenRouter), not an agent orchestrator (agent-backbone does that; the
runtime talks to it through tools), and not a user interface (it feeds
one).

## Session

A conversation with an id the client chooses. A session holds a **tree of
messages**: every message has a `parent_id`, the session tracks its active
leaf, and the path from the root to that leaf is what the model sees. A
client can therefore branch from an earlier message by sending a new
message with that message as parent.

With Postgres the tree is persisted and survives restarts; without it
sessions live in memory for the life of the process. Sessions expire after
`ASSISTANT__SESSION_TTL_HOURS` (24 by default).

## Turn

One request from the client and everything the model does in response:
thinking, tool calls, tool results, text. A turn ends with
`final_response` and then `agent_status: completed`. The non-streaming
`POST /api/chat` returns the final text; the Socket.IO transport streams
every step.

A turn is bounded by `max_turns` (agent loop iterations, 10 by default)
and by the provider's own limits.

## Message types

| `message_type` | Meaning |
|---|---|
| `standard` | A user message. Starts a turn; a running turn on the same session is cancelled and replaced. |
| `steering` | A mid-turn nudge ("focus on X"). Queued while a turn is live and delivered to the model at its next step; promoted into a normal message when no turn is running. Never cancels anything. |

A **continuation** is a `standard` message that carries `tool_call_id` and
`tool_result`: the client has executed a host tool and is handing the
result back (see host tools below). A continuation does not cancel the
turn that asked for it.

## Tools

The model can call **backend tools**: functions the runtime executes.
The full list with schemas is at `GET /api/debug/tools`. Built-in tools
are part of the runtime and present whenever their own service is
(artifacts need Postgres, media a provider key). Capabilities (what the
assistant can do) are offered only when a provider (who does it) is
configured, so the model is never given a tool that cannot work;
configuration lists the providers.

| Group | Tools | Needs |
|---|---|---|
| time, screen | `get_time`, `look_at_screen` | nothing (a host that sends screenshots, for the screen) |
| notes | `manage_notes` | `NOTES_PATH` |
| library | `list_documents`, `read_document` | `LIBRARY_PATHS` |
| artifacts | `manage_artifacts` | Postgres |
| peers, rooms, reminders, activity, workgroups, repositories | `list_agents`, `start_agent`, `send_agent_message`, `create_meeting_room`, `add_schedule_item`, `get_delivery_status`, `create_swarm`, `onboard_repo`, ... | `BACKBONE_URL` |
| approvals | `list_agent_plans`, `approve_plan`, `reject_plan` | `AGENT_STATE_DIR` |
| issues | `create_issue`, `search_issues`, `get_issue_details`, `comment_on_issue`, `close_issue` | `GITHUB_TOKEN`, `GITHUB_REPO_OWNER`, `GITHUB_REPO_NAME` |
| messaging | `respond_telegram` | `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID` |
| media | `generate_image`, `generate_video` | an image provider key; the `[video]` extra and a Runway or Luma key |
| subagent | `run_subagent` | nothing (uses the configured model) |

A tool whose call fails returns a structured error
(`{"success": false, "error": ..., "error_code": ...}`) instead of
failing the turn.

**MCP tools** come from servers listed in `mcp_servers.json` (or the file
at `MCP_CONFIG_PATH`) and are always available; the runtime starts nothing
when the file is absent.

## Host tools

Tools the **host application executes**, not the runtime: navigate to a
page, select an item, refresh a view. They are declared in configuration
(`TOOLS__HOST_TOOLS`), so the runtime ships none. When the model calls
one, the runtime emits a `tool_call` event with `category: "host"`,
ends the turn with `final_response.pending_tool_call`, and waits. The host
performs the action and sends a continuation with the matching
`tool_call_id` and a `tool_result`; the runtime resumes the model with
that result. A session holds at most one pending host tool call.

## Host context

What the host is showing right now, sent with a message (or with
`assistant:join_session` to warm the session) as `host_context`:

```json
{
  "page": {
    "name": "tasks",
    "description": "The issue list, filtered to open bugs.",
    "data": {"issues": [{"number": 42, "title": "Fix login"}]},
    "state": {"filters": {"label": "bug"}},
    "actions": [{"event_type": "select_issue", "label": "Select an issue",
                 "params": [{"name": "number", "type": "integer", "required": true}]}]
  },
  "navigation": [{"name": "agents", "description": "Running agents"}],
  "background": {"agents": {"state": "idle", "summary": {"count": 3}}}
}
```

Only `page.name` is required. `data` is rendered for the model as
key-value pairs, `state` is passed through as JSON, `actions` lists the
events the host accepts, `navigation` the places it can go, `background`
summaries of what is off screen. The last context a session received is
reused for later messages that carry none.

The page name also selects tools: `TOOLS__PAGE_SCOPES` maps a page name to
the backend tools allowed while the host shows it. Pages that are not
listed get every tool.

## Prompt artifacts

The system prompt is assembled from named texts called artifacts:

| Artifact | Role |
|---|---|
| `soul` | enduring purpose, values, non-negotiables |
| `persona` | style, stance, voice |
| `communication_protocol` | how to answer on each channel |
| `ecosystem` | the agents and systems the assistant works with |
| `scratchpad` | short-lived operational memory, editable by the model |

Default texts ship with the package. With Postgres, each artifact is
versioned: a new version is proposed, approved, or rolled back through
`/api/artifacts`, and the active version overrides the default. The
`ecosystem` artifact is the one to edit first; it is where you describe
your own environment.

After the artifacts come the connected MCP servers, the current time, the
host context and the session's **working memory**: a small structured
summary (goal, progress, next steps, key decisions) the runtime extracts
after each turn when `enable_working_memory` is on.

## History

Long conversations are compacted before they reach the model: the history
service keeps the most recent messages verbatim (`HISTORY__RETAIN_RECENT`),
protects recent tool results, and summarises older messages with the
summarisation model so the whole history fits `HISTORY__TOKEN_BUDGET`
tokens. The tree in the session is never modified by compaction; only
what is sent to the model is.

## Envelopes

Messages that arrive from somewhere other than the host carry a
provenance tag at the start of the text:

| Envelope | Origin |
|---|---|
| `[via:telegram from:<name>]` | a person on Telegram |
| `[via:tmux from:<agent>]` | an agent's terminal session |
| `[via:room room:<id> from:<sender>]` | a meeting room |
| `[via:backbone]` | a system notification from agent-backbone |
| `[via:heartbeat]` | the runtime's own periodic check-in |

The communication protocol artifact tells the model to answer on the same
channel (with `respond_telegram`, `send_agent_message` or
`send_meeting_message`). Text after an envelope is untrusted input.

## Inbox and heartbeat

Other systems can leave messages for the assistant through
`POST /api/inbox` (a note with a severity) or `POST /api/assistant/inject`
(a message delivered into a session as if a person had sent it, with an
envelope). The **heartbeat** uses the same path to inject
`[via:heartbeat]` every `HEARTBEAT__INTERVAL_SECONDS`, which gives the
model a regular chance to look at the inbox and act. Both need Postgres.

## Runtime settings

Three tiers, resolved per request: the frozen configuration from the
environment, a mutable overlay changed at runtime through
`PATCH /api/settings` (persisted when Postgres is up), and per-request
overrides in the message's `config`. The most specific wins. See
[configuration](configuration.md).
