# Persistence and recovery

What the runtime stores, what survives a restart, how a host action that
was waiting on the host is recovered, and what is deliberately not
promised. Read [concepts](concepts.md) first for sessions, turns and host
tools.

## What is stored

| State | With Postgres | Without Postgres |
|---|---|---|
| Sessions, the message tree, steering | rows in `sessions`, `messages`, `steering`; written through on every change | process memory, for the life of the process |
| The pending host action | `sessions.pending_action` | process memory |
| Prompt-artifact versions, runtime settings, provider keys, inbox | rows | memory (artifacts, settings), unavailable (keys, inbox) |
| Traces (debug) | rows | not stored |

The session store is a write-through cache: every change is written to the
row before the in-memory context is updated. The row may be ahead while a
write completes, but memory is never durably ahead of the row. A session that is not in
the cache is loaded from its rows on first use. Sessions expire after
`ASSISTANT__SESSION_TTL_HOURS`; expiry deletes the session with its messages,
steering and pending action.

The substrate is deliberately the application's own tables. The upstream
options were compared and deferred; see [the decision](#upstream-decision)
below.

## The pending host action

A host tool call ends the turn. The runtime then:

1. writes the assistant message with the call(s) and no result;
2. writes `pending_action` (`tool_call_id`, `tool_name`,
   `assistant_message_id`, `batch`: every host call of that response in
   order) on the session row;
3. emits `final_response.pending_tool_call` for the first call and waits.

When the batch has more than one call, each continuation records its
result on the assistant message and moves `pending_action` to the next
call without running the model; the last continuation resumes the model.

When a continuation arrives, the accepted result is written on the same
assistant message first. For a batch with calls still waiting, `pending_action`
then moves to the next call and the turn ends with that call as
`pending_tool_call`, without a model run; only the final continuation clears
`pending_action` and resumes the model. A crash between steps 1 and 2 leaves a call without a result and no
pending action: it is resolved as `unknown` on the next load. A crash after
the result is written but before the row is cleared leaves a pending action
whose call already has a result: the load drops it, because the result is
already recorded.

On load, the stored pending action is restored only when its assistant
message exists and the call still has no result. Every other call without a
result is given a synthetic `unknown` result so the history stays consistent.

`GET /api/sessions/{id}` returns `pending_action` with the call id, tool
name, arguments and assistant message id, so a host that reconnects or
restarts can perform the waiting action and continue without reading the
transcript.

## Action outcomes

A host tool entry in the stored segments (`GET /api/sessions/{id}/messages`)
carries the native `outcome` the model sees and, when the action did not
complete, a `status` saying how it was resolved.

| `status` | `outcome` | Produced when | External effect |
|---|---|---|---|
| completed (no `status` key) | `success` | the host sent its result | happened, per the host |
| `failed` | `failed` | the continuation carried `tool_outcome: "failed"` | did not happen, per the host |
| `cancelled` | `interrupted` | the turn was cancelled or stopped by a limit before the call returned | unknown |
| `superseded` | `interrupted` | a new user message arrived before the continuation, or core closed an older dangling call when a newer result arrived | never recorded |
| `unknown` | `interrupted` | the session was loaded without a matching pending action, or `POST /api/sessions/{id}/repair` was called | unknown |

The runtime never asserts that a `cancelled`, `superseded` or `unknown`
action had no effect. If the host knows, it says so with the continuation
(`tool_outcome`) or by telling the user.

## Reconnect, restart, duplicates, retries

- **Reconnect.** A Socket.IO disconnect leaves the turn running and its
  events are not replayed. Read the session (`messages`, and
  `pending_action` on the session) to recover the state, then continue.
- **Restart.** With Postgres, a pending host action survives: the
  continuation is accepted by the new process. Without Postgres the
  session is gone with the process, and the continuation is rejected as
  a session error because the session does not exist.
- **Duplicate continuations.** Once a call's result is recorded (whatever
  its status), any continuation for that `tool_call_id` is rejected with
  `Continuation rejected: the result for tool call '...' was already
  recorded (status: ...)`. A recorded result is never applied twice, in
  the same process or after a restart.
- **Retries.** The runtime never re-issues a host action: the model asked
  once, the host performs it once. A host that retries on its own side
  must make the action idempotent or reconcile an unknown first attempt
  itself; `tool_outcome: "failed"` is for actions the host knows did not
  occur. If
  the continuation is accepted and the model step after it fails, the
  result stays recorded and the client continues with a new message
  rather than resending the continuation.
- **Repair.** `POST /api/sessions/{id}/repair` resolves the pending action
  and every call without a result as `unknown`, in memory and on the rows,
  so a session stuck on a host that will never answer can continue.

## Worker and concurrency topology

Supported: one runtime process per set of sessions. Within a process, many
sessions run concurrently and each session has at most one live turn; a new
message cancels the live turn and waits for its cleanup, a continuation
waits without cancelling.

The in-memory context is authoritative for a session while its process
runs, and it is refreshed from the rows only on a cache miss. Within a
process, turns on one session are serialised: a continuation is planned only
after the previous turn has finished and persisted, so two continuations for
the same call arriving together are accepted once and rejected once. Two
processes serving the same session at the same time would each cache their
own copy; the runtime does not coordinate them, and each could accept the
same continuation. To run several processes against one database, route
each session to one process (sticky sessions by `session_id`). A process that restarts, or a session that moves to another
process after the first stopped, is the supported recovery path and is what
`tests/compatibility/test_recovery.py` exercises: a cold cache over the same
rows accepts the pending continuation, rejects duplicates and resolves
unanswered calls.

## Retention, branching and existing data

- Migration `0022` adds the nullable `pending_action` column; nothing else
  changes. Rows created before it have no pending action and their
  unanswered calls become `unknown` on the next load.
- Branching is unaffected: the pending action names its assistant message,
  and a branch from an earlier message is a new user message, which
  supersedes the pending action like any other.
- Retention is the session TTL; a pending action does not extend it.

## Local persistent option

The documented local persistent setup is Postgres in Docker:
`make db-up && make db-upgrade` from a checkout, or `assistant-runtime migrate`
from an installed package (the migrations ship in the wheel); see
[getting started](getting-started.md#6-optional-postgres).
It is the same schema and the same migrations as production. SQLite is not
supported: the schema uses `JSONB`, `INSERT ... ON CONFLICT` through the
Postgres dialect, Postgres server defaults and a partial unique index, and
the 22 migrations are written for Postgres. Adding SQLite would need a
second schema path and is tracked as follow-up work rather than promised.

## Upstream decision

Recorded on 2026-09-06 against Pydantic AI 2.38.0 (2.40.0 as the comparison
target, see [compatibility](compatibility.md)).

- **`pydantic_ai.durable_exec` (Temporal, DBOS, Prefect): deferred.** These
  make one agent *run* durable by journaling model requests and tool calls
  in an external workflow engine, and they require that engine and its
  worker topology. The runtime's recovery unit is different: a host action
  ends the run, and the continuation starts a new run from the persisted
  message tree. The pending action is state *between* runs, which the
  session row now holds. Upstream also documents that a durable unit may
  execute more than once after a crash, so it does not provide the
  exactly-once external effects the issue ruled out either. Applications
  that want durable model calls can attach a durability capability through
  `AssistantDefinition` (see [composition](composition.md)); the runtime
  does not adopt one.
- **Harness `StepPersistence`: deferred.** It persists harness steps, not
  the graph state of a run, and the runtime does not use the Harness loop
  (see the compaction evaluation in [compatibility](compatibility.md)).
- **Native message history: already used.** Assistant rows store
  `new_messages()` as segments and rebuild `ModelMessage`s from them; the
  native `outcome` values survive the round trip. Provider-side
  conversation ids are not used, so history does not depend on a provider.

Guarantees, in one place: sessions, messages, steering and the pending host
action are durable with Postgres; a continuation is accepted after a
restart when the matching `pending_action` row was committed, and the
unanswered call is resolved as `unknown` otherwise; a recorded result is
never applied twice; unanswered calls are
distinguishable as `cancelled`, `superseded` or `unknown`; external effects
are never assumed, never undone and never retried by the runtime.
