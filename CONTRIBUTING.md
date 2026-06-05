# Contributing

This is a research-grade project under active development; interfaces may change.
Issues, ideas, and PRs are welcome, especially around retrieval quality, the eval
metrics, and additional model backends.

## Dev setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows; use bin/activate on *nix
pip install -e ".[dev]"
cp .env.example .env                               # then fill in keys
pytest -q          # pure metric + smoke tests, no network
ruff check .
```

The vector store and full eval runs need Postgres/pgvector (`docker compose up -d`)
plus a corpus ingest; see the README. The unit tests need neither.

## Bar for changes

- **Tests required.** New metric or pipeline logic gets a test; bug fixes get a
  regression test. Keep the unit tests network-free and deterministic.
- **Lint clean** (`ruff check .`).
- **Keep the eval axes distinct.** Retrieval quality, faithfulness, hallucination
  rate, citation accuracy, and abstention correctness are measured and reported
  separately. Do not collapse them.
- **Grounding is the point.** Changes that let the agent answer without cited
  support, or that weaken abstention, need a strong justification and eval numbers.
- Keep the medical framing intact: this is evidence retrieval, not medical advice.

## Scope

See [SPEC.md](SPEC.md) for the milestone plan. New model backends and benchmark
runners are welcome.
