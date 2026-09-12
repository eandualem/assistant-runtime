# Reference application and adoption status

[Design Studio](https://github.com/eandualem/design-studio) is the reference
browser application: a Markdown and Mermaid editor with an assistant that
reads page context and edits document blocks through host actions. It uses
the public Socket.IO, HTTP and `host_context` contracts. It is a separate
application, with its own domain profile and no runtime source edits.

## Run the application

Follow Design Studio's [installation instructions](https://github.com/eandualem/design-studio#running-it).
Clone that repository first so its `profiles/design-studio.toml` is available,
install the released runtime with `uv tool install assistant-runtime`, and
set `ASSISTANT__PROFILE` to the profile's absolute path before starting the
runtime. The studio uses port 7130 and connects to the runtime on port 7100
by default. Its README describes provider configuration and URL overrides.

Design Studio's [demo](https://github.com/eandualem/design-studio#the-demo)
is an opt-in live model exercise, separate from its deterministic tests.
It records timings, actions, errors, tokens and estimated costs. It includes
scenarios for attachments, branching, cancellation, host-action reload and
artifact proposals. A scenario in the runner is not evidence that it passed.

The runtime also has a minimal [AG-UI browser example](https://github.com/eandualem/assistant-runtime/tree/develop/examples/agui),
and [composition](composition.md) documents a store assistant definition.
These illustrate public contracts; they are not additional independently
validated adoption exercises. Socket.IO remains Design Studio's transport.

## Available behavior and validation limits

The runtime supports configured profiles, host context/actions, message trees,
cancellation, artifact mutation policies and usage accounting. Postgres adds
persistent sessions and pending actions; without it those are process-local.
See [persistence](persistence.md) for the exact recovery guarantees and
[compatibility](compatibility.md) for deterministic execution coverage.

As of 2026-09-12, Design Studio has documented package installation and demo
commands. Its retained September 7 run against runtime 0.1.0 created a document
and edited blocks successfully. The run's parse-repair scenario did **not**
produce a render error, so it does not establish repair. Older feature exercises
and current implementation support do not constitute a fresh complete live run.

[Issue #95](https://github.com/eandualem/assistant-runtime/issues/95) tracks the
remaining acceptance evidence: a second unrelated assistant definition run,
three unrelated integration exercises including an outside developer, measured
onboarding assistance/time, and complete recovery, rejected-action, artifact
and cost demonstrations. An agent-run test is not an outside-human adoption
trial. The original issue's non-browser example conflicts with a later scope
exclusion and remains a recorded scope discrepancy, not a shipped example.
The 15-minute installation and one-hour first-action targets are experimental
targets, not product guarantees.

## Separately runnable runtime smoke checks

From a runtime checkout, install development dependencies with
`uv sync --locked --extra dev`. The `smoke/` directory is outside the default
`tests/` collection. Running `uv run pytest smoke` without the opt-in variables
skips both probes; a skip is not a live pass. `make check` remains independent
of running services and credentials.

### Model and public HTTP path

Start a dedicated runtime with a provider configured as in
[getting started](getting-started.md). For this minimal probe, select the
neutral profile and disable application tools, then use an unused port:

```bash
ASSISTANT__PROFILE=neutral TOOLS__BUILTIN_TOOLS='[]' \
TOOLS__PROVIDER_CAPABILITIES='[]' \
uv run assistant-runtime serve --port 7101 --no-replace
```

In another terminal, run:

```bash
RUN_MODEL_SMOKE=1 SMOKE_RUNTIME_URL=http://127.0.0.1:7101 \
uv run pytest smoke/test_model.py -q --junitxml=/tmp/model-smoke.xml
```

This makes a real model request and may incur provider charges. The probe
expects a short marker reply, reads the saved assistant message through HTTP,
records model/elapsed time/stored usage in the JUnit report, and deletes its
own UUID-named session after a successful response. If the POST fails or times
out, it requests cancellation and leaves the session UUID for inspection; it
does not claim cleanup finished while a server turn may still be draining.
Use a dedicated trusted-local test instance. It does
not measure onboarding, tool behavior, UI rendering or provider billing accuracy.
No provider key means this check has not been run, even if deterministic CI passes.

### Database commit and reconnect

Configure a scratch Postgres database using `DATABASE__*`, then migrate it:

```bash
uv run assistant-runtime migrate
RUN_DATABASE_SMOKE=1 uv run pytest smoke/test_database.py -q
```

This probe needs no provider or HTTP server. It writes a uniquely named session
and message, commits, closes the writer's connection pool, and verifies the
content and JSONB data through a fresh database service. It deletes its own
session afterward (message deletion cascades). An unreachable database fails
instead of accepting the runtime's normal in-memory fallback. The probe tests
the real schema/commit/reconnect boundary; it does not simulate a process crash
or validate the host continuation protocol. Run the reference app recovery
scenarios separately with Postgres configured to exercise those paths.

## Record an integration exercise

For each unrelated application, record these fields with a date and exact
application/runtime versions in an issue or shareable evidence document:

- Participant role: original project developer, another agent, or outside
  developer; do not infer independence from a separate repository.
- Domain, assistant profile, host transport and the task to complete.
- Setup start/end and first useful domain-action time; list required assistance
  and every runtime source edit, including zero when measured.
- Task result, recovery/failure scenarios actually triggered, latency, model,
  tokens and cost source; label unavailable values and estimates explicitly.
- Reproduction commands and redacted logs/screenshots. Record failed attempts
  as well as successful ones; remove credentials and private host content.

Keep model and database smoke results separate from adoption measurements.
A failed setup, an untriggered recovery scenario or an unmeasured field must
remain visible rather than being counted as completed acceptance.
