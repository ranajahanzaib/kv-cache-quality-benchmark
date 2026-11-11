# KV Cache Quality Benchmark

**Measure what KV-cache compression costs your model.**

A local, reproducible test harness for comparing answer correctness, long-context fact retention, and memory use across llama.cpp KV-cache configurations. It runs against your GGUF and llama.cpp installation; it does not download models or call cloud services.

## What it measures

KV Cache Quality Benchmark sends the same serialized prompt to the same model artifact under F16, Q8, Q4, and mixed K/V cache formats. It records tokenizer IDs, short-answer scores against fixture-defined ground truth, llama.cpp timing metadata when available, and sampled process RSS. Unsupported formats and invalid inputs are excluded from accuracy scores. The F16 result is a reference point, not an oracle: absolute ground-truth accuracy and baseline-conditioned retention answer different questions.

The fixture suite includes needle retrieval, independent critical-fact questions, similar-ID distractors, multi-fact questions, and unanswerable prompts. Each complete serialized prompt is resized to a 72% context target within a 5% tolerance using the selected model's tokenizer; actual token count and fact position are stored. Unreachable targets and prompt-plus-output overflow are marked invalid. Results from small synthetic samples are exploratory.

## Requirements

- Python 3.12 or newer and [uv](https://docs.astral.sh/uv/)
- A local `llama-server` build and user-provided GGUF model
- macOS Apple Silicon is the primary target; Linux is expected to work when the installed llama.cpp build exposes the required options

No CUDA, Docker, hosted service, or bundled model weights are required. The selected llama.cpp build determines the inference backend; KV Cache Quality Benchmark records backend information when the build reports it and does not select a backend itself. For macOS Metal, build the official [llama.cpp project](https://github.com/ggml-org/llama.cpp) with its documented Metal support. KV Cache Quality Benchmark inspects the binary's own `--help` and `--version` output.

## Install

```bash
git clone https://github.com/ranajahanzaib/kv-cache-quality-benchmark.git
cd kv-cache-quality-benchmark
uv sync --locked
```

Point the tool at a compatible GGUF and server executable (or put `llama-server` on `PATH`). A concrete smoke-test artifact is Qwen's official `Qwen2.5-3B-Instruct-GGUF` `qwen2.5-3b-instruct-q4_k_m.gguf` (about 2.1 GB):

```bash
mkdir -p models
curl -L -o models/qwen2.5-3b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
shasum -a 256 models/qwen2.5-3b-instruct-q4_k_m.gguf
# Expected SHA-256: 626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d
```

The model is under the [Qwen Research License Agreement](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/blob/main/LICENSE), which restricts use to non-commercial purposes unless separately licensed. Review that license before use. The model repository provides the file and checksum; KV Cache Quality Benchmark does not bundle or fetch weights.

For an Apple Silicon source build, llama.cpp enables Metal by default on macOS:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release -j 8 --target llama-server
```

See the [llama.cpp build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for current platform options. Alternatively install a packaged build, then check its options with `uv run kv-cache-quality-benchmark doctor`.

## Run

```bash
uv run kv-cache-quality-benchmark doctor
uv run kv-cache-quality-benchmark probe --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf
uv run kv-cache-quality-benchmark run --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --preset smoke
uv run kv-cache-quality-benchmark generate --model ./models/qwen2.5-3b-instruct-q4_k_m.gguf --preset smoke --seed 42
uv run kv-cache-quality-benchmark report <experiment-id>
```

Presets: `smoke` uses a 2K context, three K/V configurations, and one generated case per workload; `standard` uses 2K, 4K, and 8K contexts with F16, Q8, and Q4; `extended` uses 2K, 4K, 8K, and 16K contexts with all four built-in configurations and three repetitions. Standard generates five cases per workload and extended generates twelve. Override contexts, fixture count, selected config IDs, repetitions, and GPU layers with CLI options. `--help` lists all commands and arguments. Requested contexts must fit the model's supported context; unsupported lengths are reported.

The CLI writes results to `~/.kv_cache_quality_benchmark/experiments.sqlite3` by default. Export a saved run with its printed ID:

```bash
uv run kv-cache-quality-benchmark export <experiment-id> --output ./reports
```

This produces JSON, CSV, and Markdown. The JSON includes individual prompts, expected answers, answers, scores, timing data, token IDs, and unsupported configurations.

## Dashboard

The CLI does not require Node.js. Building the dashboard requires Node.js 20 or newer and pnpm.

For UI development, run the API and Vite in separate terminals. In terminal 1:

```bash
uv run kv-cache-quality-benchmark dashboard
```

In terminal 2:

```bash
cd frontend
pnpm install
pnpm run dev
```

Open the Vite URL printed in terminal 2. It proxies API requests to `127.0.0.1:8000`.

To serve the bundled UI after building it:

```bash
(cd frontend && pnpm install && pnpm run build)
uv run kv-cache-quality-benchmark dashboard
```

The dashboard browses saved results and clearly labels process RSS; it does not claim RSS is KV allocation. Its API has no authentication and binds to loopback by default; do not expose it to an untrusted network.

Use a checked-in YAML preset with `uv run kv-cache-quality-benchmark run --config configs/smoke.yaml`. CLI arguments override preset values. The runner checkpoints completed trials to SQLite and JSONL, so interruption preserves finished attempts.

## Scoring and interpretation

Identifier answers use conservative exact matching with whitespace, quote/punctuation wrapper, and case normalization. Contradictory prose is not accepted. Unanswerable questions require `NOT_FOUND` (or a narrow explicit abstention) and a fabricated value is an unsupported assertion. Pairing distinguishes preserved successes, regressions, improvements, persistent failures, and unavailable comparisons.

Configured context capacity and actual prompt token count are separate. llama.cpp `/apply-template` serializes messages with the loaded model's chat template. The server tokenizer produces token IDs, and the runner checks identical IDs across configurations. Inputs that exceed context plus generation budget are skipped rather than silently truncated. Fact positions are measured from the serialized prompt prefix in token space. Context-specific fixtures preserve the same underlying facts while changing fictional filler length.

RSS is a point sample of process-resident memory and includes more than KV storage. KV allocation remains unavailable unless llama.cpp exposes a trustworthy measurement. Timing fields vary by llama.cpp build. Greedy decoding can still vary across hardware/builds. A result applies only to the recorded model hash, runtime, hardware, fixture set, and settings. Do not generalize a small sample or call a cache format safe based on no observed failures.

## Development

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check src tests
uv run mypy src/kv_cache_quality_benchmark
```

CI does not download model weights or run expensive inference. For design details see [methodology](docs/methodology.md), [metrics](docs/metrics.md), [limitations](docs/limitations.md), and [reproducibility](docs/reproducibility.md). KV Cache Quality Benchmark is MIT licensed. [llama.cpp is separately MIT licensed by the ggml authors](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE); model weights remain under their own licenses. Check the terms for the model you use.

## Contributing and security

See the [contribution guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before opening an issue or pull request. Report vulnerabilities using the process in the [security policy](.github/SECURITY.md), not in public issues.
