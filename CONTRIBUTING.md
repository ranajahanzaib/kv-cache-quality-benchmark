# Contributing

Contributions that improve correctness, reproducibility, documentation, or usability are welcome.

## Development setup

Install the Python development dependencies and dashboard dependencies:

```bash
uv sync --locked --extra dev
(cd frontend && pnpm install)
```

Run the local checks before opening a pull request:

```bash
uv run ruff check src tests
uv run mypy src/kv_cache_quality_benchmark
uv run pytest
cd frontend && pnpm run build
```

Tests use mocked runtime responses and do not download models. Real inference checks require a
locally available `llama-server` and GGUF model; do not add model weights or generated benchmark
results to pull requests.

## Changes and pull requests

Open an issue for substantial changes so the scope can be discussed. Pull requests should explain
the user-visible change, include focused tests for behavior changes, and update documentation when
commands or output change. Include relevant check results and note any checks that could not run.

Never include API keys, credentials, private data, or unredacted logs in issues or pull requests.
