# The host contract

What an application tells the runtime about itself, and what it gets
back. Version 1. The Python source of truth is
`assistant_runtime/host_context.py`; this page describes the wire form.

Everything a host sends here is *application context*: it shapes the
prompt and the tools of a turn. None of it is identity or authorization.
A client cannot widen what it may do by describing itself differently;
trusted identity is a separate concern.

## `host_context`

Sent with a message (`POST /api/chat`, Socket.IO `assistant:message`) or
with `assistant:join_session`. Keys may be camelCase or snake_case. The
runtime stores the validated, snake_case form on the session and reuses
it for later messages that carry none.

```json
{
  "version": 1,
  "host": {"name": "web-dashboard", "kind": "browser", "version": "2.1"},
  "view": {
    "name": "tasks",
    "description": "The issue list, filtered to open bugs.",
    "data": {"issues": [{"number": 42, "title": "Fix login"}]},
    "state": {"filters": {"label": "bug"}}
  },
  "navigation": [{"name": "agents", "description": "Running agents"}],
  "actions": [
    {
      "name": "select_issue",
      "description": "Select an issue in the list.",
      "parameters": {"type": "object", "properties": {"number": {"type": "integer"}},
                     "required": ["number"]}
    }
  ],
  "attachments": [{"kind": "image", "purpose": "screenshot", "data_uri": "data:image/png;base64,..."}],
  "background": {"agents": {"state": "idle", "summary": {"count": 3}}},
  "captured_at": "2026-09-05T10:00:00Z",
  "extensions": {"tenant": "acme"}
}
```

| Field | Meaning |
|---|---|
| `version` | `1`. Another value is rejected with a message naming the supported version. |
| `host` | Descriptive metadata: `name`, `kind` (`browser`, `mobile`, `desktop`, `terminal`, `service`, `other`), `version`. Rendered for the model; never trusted for anything. |
| `view` | What is on screen: `name` (required within `view`; also selects `TOOLS__PAGE_SCOPES`), `description`, `data` (curated content, rendered as key-value pairs), `state` (selection, filters, workflow state; passed as JSON). `page` is accepted as an alias. |
| `navigation` | Places the host can go: `[{"name", "description"}]`. |
| `actions` | Actions the host performs when the model calls them, for this turn. Each becomes a host tool: `name` (tool-name rules), `description`, `parameters` (a JSON schema of type object). See below. |
| `attachments` | Content the host attaches to the context; the same shape as message attachments below. |
| `background` | Summaries of what is off screen, `{"name": {"state": "...", "summary": {...}}}`; rendered as one line. |
| `captured_at` | When the host captured the context. The prompt says how old it is; contexts older than an hour are described as stale. Nothing is rejected for age. |
| `extensions` | Host-specific data the runtime does not interpret; shown to the model as JSON. The extension point. |

Unknown fields anywhere are rejected (`422` on HTTP, `assistant:error` of
type `validation` on Socket.IO) with the field path. `view.data` +
`view.state` + `extensions` may total 32,000 characters of JSON; a context
carries at most 50 navigation targets, 32 actions and 16 attachments.
Curate what the model needs instead of sending everything. Keys inside
`view.data`, `view.state`, `background`, `extensions` and action
`parameters` are the host's own and are passed through verbatim; only the
contract's field names are normalised.

Absent context: the prompt has no host section and every backend tool is
available. A context without `view` (a service, a background worker) is
fine: it can still carry `host`, `actions` and `extensions`.

## Attachments

On the message body as `attachments`, or inside `host_context`; both reach
the turn the same way (message-level first, duplicates dropped):

```json
{"kind": "image", "purpose": "reference", "name": "photo", "description": "the receipt",
 "data_uri": "data:image/jpeg;base64,..."}
```

| Field | Meaning |
|---|---|
| `kind` | `image` (default), `document`, `text` |
| `purpose` | `reference` (default): the content goes into the model's message as native Pydantic AI content (`BinaryContent`, `ImageUrl`, `DocumentUrl`, or text). `screenshot`: an inline image the model only sees when it calls `look_at_screen`. |
| exactly one of `data_uri`, `url`, `text` | the content; a data URI or text may be up to 12,000,000 characters |
| `name`, `description`, `media_type` | optional labels |

Legacy `images` (data URIs) and a top-level `screenshot` field become
screenshot attachments. A malformed attachment is a `422`. Attachments are
seen in the turn they arrive with; they are not replayed from history in
later turns, and media is not stored in sessions.

## Actions

Two ways to declare what the host can do; both use the same protocol:

- **Configured host tools** (`TOOLS__HOST_TOOLS`), available every turn.
- **Request-declared actions** (`host_context.actions`), available for
  the turns that carry them (a session reuses its last context). A name
  that collides with a registered tool is ignored with a warning.

When the model calls one, the runtime emits `tool_call` with
`category: "host"` and a `call_id`, ends the turn with
`final_response.pending_tool_call` (`tool_name`, `call_id`, `arguments`),
and waits. The host performs the action and sends a continuation: the
same message body with `tool_call_id` = the pending `call_id` and
`tool_result` = any JSON. The run resumes with that result on the same
assistant message. Conventions for the result: an object with `error`
tells the model the action failed; a screenshot data URI anywhere in it is
removed from what the model reads and offered to `look_at_screen`
instead.

Errors: a continuation for a call that is not pending (or after a newer
message abandoned it) is rejected as a session error (`409` on HTTP); a
session holds at most one pending call.

## Versioning

`version` is the contract version. Additive changes (new optional fields,
new enum values) stay within version 1; a change that alters the meaning
of an existing field bumps it, and a runtime rejects versions it does not
speak. Clients should send the version they were written against.
