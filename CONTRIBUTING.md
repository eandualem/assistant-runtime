# Contributing

Thanks for taking a look. This is a young project; issues and pull requests
are welcome.

## Setup

Python 3.12 or 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked --extra dev     # runtime plus the test and lint tools
make check                       # ruff check + format check + pytest
```

The suite needs no API key, no Postgres and no network. `make check` is the
gate: it must pass before you push. `make fix` applies the formatter.

## Making a change

- Branch from `develop` and open the pull request against `develop`. `main`
  only receives merges from `develop`.
- Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`) and say *why* in the body.
- Update the documentation page that describes the behaviour you changed in
  the same pull request. The pages live in `docs/` and ship inside the wheel.
- Add tests next to the ones for the area you touched. Database code is
  exercised through fakes, and the model boundary is mocked; the cases in
  `tests/compatibility/` run real Pydantic AI execution against a scripted
  model with provider requests disabled.

`AGENTS.md` documents the architecture and the invariants the code holds to.
Read it before changing anything structural: module layout, startup order,
configuration tiers, the tool registry, the session model or the turn
pipeline.

## Reporting a problem

Open an issue with what you ran, what happened and what you expected. For a
runtime problem, `assistant-runtime doctor` output helps. Please do not paste
API keys or tokens.
