# Pydantic AI compatibility

The runtime supports Pydantic AI 2.x starting at 2.38.0 (`>=2.38.0,<3`).
The release/development baseline remains **2.38.0 in `uv.lock`**. Version
2.40.0 is the comparison target for the migration work in
[#84](https://github.com/eandualem/assistant-runtime/issues/84). The upper
bound prevents an unreviewed major upgrade; it does not assert that every
future 2.x release has been tested. Prefer the lockfile for deployments.

## Evidence and reproduction

The 2026-09-05 baseline uses Python 3.12.14. The execution, history and native
composition cases in `tests/compatibility/` pass on both 2.38.0 and 2.40.0. They run real `Agent`
execution against `FunctionModel`; provider requests are disabled. Session
reload replaces the database persistence boundary with a fake, so this is
not a Postgres restart test or a live provider test.

```bash
uv sync --locked --extra dev
uv run pytest tests/compatibility -q
uv run --no-sync --with pydantic-ai-slim==2.40.0 --with pydantic-graph==2.40.0 \
  python -m pytest tests/compatibility -q
make check
```

The `--with` command creates an isolated overlay; it does not upgrade the
project lock or installed environment. CI runs the compatibility cases on
both versions and keeps the existing full gate on Python 3.12 and 3.13.
Update the comparison target deliberately when reviewing a dependency
upgrade, then review the lockfile diff and run the full gate.

| Component | Tested versions / requirements | Evidence boundary |
|---|---|---|
| Core execution | `pydantic-ai-slim` and `pydantic-graph` 2.38.0, 2.40.0 | Real-agent compatibility cases |
| Vercel UI adapter | `pydantic-ai-slim[ui]`; `starlette>=0.46.2` | `VercelAIAdapter` imports on both core versions |
| AG-UI adapter | `pydantic-ai-slim[ag-ui]`; `ag-ui-protocol>=0.1.10`, `starlette>=0.46.2` | `AGUIAdapter` imports on both; smoke environment resolved protocol 0.1.22, Starlette 1.6.0 |
| Harness | Separate `pydantic-ai-harness==0.29.0`, requiring core `>=2.38.0` | `SummarizingCompaction`, `Memory`, `StepPersistence` imports on both; no behavior-equivalence claim |
| Durable backends | Separate extras; e.g. `temporalio>=1.24.0`, `dbos>=2.10.0`, `prefect>=3.7.5` | Dependency metadata only; no backend installed or exercised |

Adapters and Harness are not runtime dependencies. Their CI import check
uses a separate overlay. Successful imports establish dependency
compatibility, not integration with sessions, authorization, or streaming.
Requirements were inspected in the published
[core](https://pypi.org/pypi/pydantic-ai-slim/2.38.0/json) and
[Harness](https://pypi.org/pypi/pydantic-ai-harness/0.29.0/json) metadata.

## Decisions supported by the cases

Test names below refer to `tests/compatibility/`. The composition work in
[#85](https://github.com/eandualem/assistant-runtime/issues/85) adopted public
event streaming and steering enqueue so native middleware wraps execution.
[#86](https://github.com/eandualem/assistant-runtime/issues/86) adds native
cancellation snapshots and application-owned finalization.

| Area | Decision and public API | Executable evidence / remaining application behavior |
|---|---|---|
| Streaming graph traversal | **Composed** `Agent.run_stream_events()` | `test_runtime_stream_order_and_complete_tool_arguments`, `test_public_stream_has_complete_tools_and_one_trailing_result`, and native capability guard/wrapper/event-processor cases; retain event naming, segment IDs, full arguments, one final/completed envelope, and suppression of repeated final text |
| Steering | **Composed** `RunContext.enqueue(priority="asap")` and `EnqueuedMessagesEvent` with the session queue | Same-run delivery remains once; `test_interrupted_steering_remains_pending_for_the_next_turn` verifies retry after cancellation before the model call or during a partial response. Acknowledge only after the consuming model request succeeds |
| Cancellation | **Composed** `CancellationToken`, `RunCancelled.new_messages()` / `.usage`, and `RunCancelled.from_cancellation()` | Native cases cover partial responses and tool-task teardown; the runtime saves snapshots through the same assistant-row persistence path, including accepted host results, and completes one cancellation lifecycle |
| Deferred host actions | **Keep composing** `ExternalToolset`, `DeferredToolRequests`, `DeferredToolResults` | Host continuation executes a backend tool once and keeps the same assistant message; batch calls resume upstream after JSON serialization, while the current host protocol rejects two pending calls |
| Tool arguments | **Compose** `ToolCallPart.args_as_dict()` | Five serialization cases plus a real streamed host continuation cover dicts, JSON objects, malformed/non-object JSON and empty args |
| Model-input history cleanup | **Replace overlapping repair** through the public `Agent` run pipeline; do not import cleanup helpers | `test_public_history_closes_dangling_calls_including_malformed_args`, `test_public_history_drops_orphaned_results_before_model_request`; preserve the source conversation and pending frontier semantics |
| Custom history selection | **Compose** `ProcessHistory` capability | `test_public_process_history_can_filter_model_input_without_editing_source`; the application owns which branch and records supply history |
| Session storage and reload | **Keep** application tree/steering persistence | `test_session_reload_restores_tree_and_repairs_unfinished_host_action`; flat upstream message serialization does not encode this tree or recover its pending host-action state |
| Compaction / working memory | **Keep pending evaluation** of Harness `ClearToolResults`, `SummarizingCompaction`, `Memory` | Existing `tests/unit/services/history/` covers current behavior; imports alone do not prove equivalent summaries, working-memory extraction or source-history retention |
| UI protocols | **Evaluate composition** with `AGUIAdapter` / `VercelAIAdapter` | Import smoke checks only; protocol-to-session mapping remains #93 |
| Definitions / evolving artifacts | **Keep** versioned application definitions; compose native instructions/capabilities | No replacement of definition, activation, rollback or mutation policy established; existing artifact tests remain authoritative |
| Usage, models, images and toolsets | **Keep composing** native `UsageLimits`, model/provider APIs, `BinaryContent`, `FunctionToolset` | Existing tests under `tests/unit/services/` and screenshot tests remain regression coverage; this baseline does not justify replacing the subscription transport or extending accounting |

The public API references are
[agent execution](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/),
[run control](https://pydantic.dev/docs/ai/api/pydantic-ai/run/),
[deferred tools](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/), and
[capabilities](https://pydantic.dev/docs/ai/capabilities/overview/).
Decisions above use the executable cases against the stated versions, since
the live documentation can advance beyond the lockfile.

## Observed boundaries and remaining coupling

- Streamed providers can keep tool arguments as a JSON string. The old
  serializer converted those arguments to `{}`, and pending host payloads
  preferred that persisted value. The baseline fixes this with native
  argument parsing. Invalid JSON retains the upstream `INVALID_JSON`
  marker instead of silently becoming an empty object.
- Core's public request pipeline both removes orphaned results and repairs
  dangling calls. Inspecting only its dangling-call helper understates that
  support. The tested malformed-history examples are not a guarantee that
  every provider will accept arbitrary damaged history.
- A turn owns a fresh native cancellation token across its agent runs.
  Cancellation retains raw native messages and usage, not reconstructed
  display deltas; host event processors can transform that display. Partial
  text and completed tools are saved, unresolved calls receive an interrupted
  outcome, and accepted host results remain on the existing assistant row.
  External task cancellation still propagates after cleanup. A timeout uses
  upstream's attached snapshot while keeping its timeout error classification.
- `cancel_session` also interrupts setup. Iterator closure/cancellation and
  shutdown cancel and drain the producer; a Socket.IO disconnect leaves it
  running. Ordinary replacement waits for cleanup, and continuations wait
  without cancelling. These are process/session guarantees, not durable run
  checkpoints or proof that cancelled external work had no effects.
- Queued steering is acknowledged after a successful model response consumes
  it. Application message segments do not retain its native `UserPromptPart`,
  so insertion into the native queue alone is insufficient: interrupted
  delivery stays pending and can be retried on the next turn.
- Session reload restores message branches and queued steering. Unanswered
  host calls are marked stale because pending state is memory-only. This
  does not prove the external action failed or prevent its late execution.
  [StepPersistence](https://pydantic.dev/docs/ai/harness/step-persistence/)
  is not a full graph checkpoint or an exactly-once side-effect guarantee.
- Runtime execution uses `Agent.run_stream_events()` without private graph
  imports or state access. A per-run capability uses public node predicates
  and `RunContext.enqueue` to deliver mid-tool steering. Screen retention
  sanitizes public `RunContext.messages` after the first model request that
  consumes the image; this remains application policy. Real cases verify
  one-request image retention and request context across native tasks.
- `assistant_record_to_flat_messages` still arranges completed and pending
  calls around upstream deferred-resumption behavior. Keep the real mixed
  backend/host continuation case green when changing it.

No dependency upgrade is required for these supported replacements. Remaining
history and recovery work is tracked by
issues [#87](https://github.com/eandualem/assistant-runtime/issues/87) and
[#92](https://github.com/eandualem/assistant-runtime/issues/92).
