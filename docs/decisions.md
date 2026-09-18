# Typed decisions

The decision capability lets an application ask typed questions about its
own state and act on typed answers, without a language model in the loop.
The runtime holds the provider key, validates the call, sends every question
to the provider in one request and returns the answers with calibrated
probabilities. It is a separate capability from chat: nothing is generated,
nothing is parsed, and the assistant profile's prompt is not involved.

The provider is [TypeSafe's Jev](https://docs.typesafe.ai), a System One
decision model served at `POST /v1/systemone`. One call carries the state and
a map of questions; the model evaluates them in parallel and answers in one
forward pass, in the provider's documented 70 to 500 ms. Batching is the point:
an application that asks eight questions per event asks them in one call.

## Enable

```bash
export TYPESAFE_API_KEY=...       # server environment only
uv run assistant-runtime serve
```

Nothing else is needed; there is no extra to install. `GET /api/decisions/status`
reports `healthy`, `configured` (the key variable is set), `provider` and `model`.
`configured` means a key is present, not that access has been tested.

The data path is explicit. Without the key, `configured` is `false` and a call
returns `503` with `Decisions need TYPESAFE_API_KEY on the runtime; no fallback
is used`. The runtime never answers a decision from a language model and never
guesses. `DECISIONS__API_KEY_ENV` names a different variable; `DECISIONS__MODEL`
selects the provider model (default `jev-latest`); see
[configuration](configuration.md#decisions-decisions__).

As documented on September 18, 2026, TypeSafe charges $0.042 per million input
tokens and nothing for output; see the current [models page](https://docs.typesafe.ai/models)
for pricing, rate limits and the per-request token budget.

## The call

`POST /api/decisions` uses the runtime's normal authentication. It needs no
session and reserves nothing.

```json
{
  "state": {
    "conversation": [
      {"speaker": "assistant", "text": "Hi there."},
      {"speaker": "user", "text": "can you clap", "transcript": "partial, still speaking"}
    ],
    "body": "at rest"
  },
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Is the latest user line an explicit request for a movement?",
      "criteria": {
        "explicit": {"what": "asks for a movement", "examples": ["can you wave", "clap for me"]},
        "incidental": {"what": "mentions a movement in passing"},
        "none": null
      }
    },
    "start": {
      "type": "noul",
      "instructions": "Should a new gesture start now?",
      "criteria": {"true": {"examples": ["wave please"]}, "false": {"examples": ["hello there"]}}
    },
    "energy": {
      "type": "score",
      "instructions": "How much energy does the moment call for?",
      "criteria": ["still", "calm", "lively", "playful"]
    }
  },
  "profile": "avatar"
}
```

`state` is any JSON the questions are about, at most `DECISIONS__MAX_STATE_BYTES`
serialised (default 256 KiB). `questions` is a map from the application's own
ids to typed questions, at least one and at most `DECISIONS__MAX_QUESTIONS`
(default 32). The three types follow the provider's
[primitives](https://docs.typesafe.ai/primitives):

- `choice` picks one option from `criteria`, a map of at least two options.
  An option's value is `null`, a description, or `{"what", "not_for", "examples"}`.
- `score` picks one level from `criteria`, a list of at least two levels, each
  a description or a structured entry.
- `noul` answers whether a statement is true. `criteria` is optional:
  `{"true": ..., "false": ...}` with the same entry shape.

`instructions` is text, or the provider's structured form. The runtime checks
the structure; the provider judges the meaning, and its rejection comes back as
`422`. Optional `profile` names a registered assistant profile, validated like
voice creation (`422` for an unknown name). One runtime key serves every
profile; the profile does not change the questions or the provider.

The response keeps the application's ids:

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "intent": {"type": "choice", "choice": "explicit", "probabilities": {"explicit": 0.86, "incidental": 0.09, "none": 0.05}, "confidence": 0.81},
    "start": {"type": "noul", "noul": 0.92},
    "energy": {"type": "score", "score": 2, "legend": {"0": "still", "1": "calm", "2": "lively", "3": "playful"}, "probabilities": {"0": 0.05, "1": 0.2, "2": 0.6, "3": 0.15}, "confidence": 0.6}
  },
  "usage": {"input_tokens": 812, "output_tokens": 0},
  "timing": {"total_ms": 356, "provider_ms": 349}
}
```

Every question has an answer of its own type; a response that lacks one, or
answers with the wrong type, is a `502`, and answers for ids that were not asked
are dropped. Fields the provider adds to an answer are passed through. `usage` and `model` are the provider's own. `timing` is
measured by the runtime for this call: `provider_ms` is the provider round trip
and `total_ms` adds the runtime's validation and answer checking. Thresholds
belong to the application; the runtime applies none.

## Errors

Every error carries `detail`. A provider error also carries
`provider_status_code` and, when the provider gave a reason, `provider_detail`
(its `detail`, `error` or `message` text, at most 500 characters), so a
rejected question can be fixed from the application's own log line. The state
and the key are never echoed by the runtime.

| Status | Meaning |
|---|---|
| `422` | The body failed validation, `profile` is unknown, too many questions, the state is too large, or the provider rejected the state or questions (`provider_status_code: 422`) |
| `429` | The provider's rate limit (`provider_status_code: 429`); retry with backoff |
| `502` | The provider rejected the runtime's key (`401`/`403`), failed, was unreachable, or answered in an unexpected shape |
| `503` | The key is not configured, or the service is not started |
| `504` | The provider did not answer within `DECISIONS__TIMEOUT_SECONDS` (default 10) |

The runtime does not retry. A consumer that asks on every transcript fragment
should keep at most one call in flight and send the latest state on the next.

## Latency

The consumer calls this while the user is still speaking, so the runtime's
share of the round trip is measured, not assumed. One `httpx` client is opened
at startup and shared, so each call reuses the provider connection.

Measured on September 18, 2026 with an 8-question, 3.8 KB request (a choice over
22 options, a choice over 15, two scores, four nouls) against a local stand-in
for the provider that answers instantly, 300 calls over localhost HTTP on a
laptop:

| | p50 | p95 |
|---|---|---|
| Client to runtime to stand-in and back | 1.0 ms | 1.2 ms |
| Client to stand-in directly | 0.3 ms | 0.3 ms |

The runtime adds about 1 ms per call to the provider's round trip; the first
call after startup added 4 ms for connection setup. This measures the runtime,
not TypeSafe: real calls are the provider's 70 to 500 ms plus network distance
to it. `timing` in every response shows both numbers for the actual deployment.

## Validation boundary

Offline tests exercise the request and answer shapes, the status mapping, the
missing-key error, profile validation and the shared client's lifecycle against
a fake provider endpoint. They do not establish provider access, answer quality
or billing; a call with a real key is the live acceptance step.
