"""SQLite persistence and portable report exports."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

from .metrics import summarize
from .models import Experiment, InferenceResult

DEFAULT_DB = Path.home() / ".kv-cache-quality-benchmark" / "experiments.sqlite3"


class Store:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS trials (experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE, fixture_id TEXT NOT NULL, config_id TEXT NOT NULL, execution_order INTEGER NOT NULL, attempt INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(experiment_id, fixture_id, config_id, attempt))")
            db.execute("CREATE INDEX IF NOT EXISTS trials_lookup ON trials(experiment_id, config_id, fixture_id)")

    def save(self, experiment: Experiment) -> None:
        payload = experiment.model_dump_json()
        with sqlite3.connect(self.path) as db:
            db.execute("PRAGMA foreign_keys=ON")
            db.execute("INSERT OR REPLACE INTO experiments VALUES (?, ?, ?)",
                       (experiment.id, experiment.created_at, payload))
            db.execute("DELETE FROM trials WHERE experiment_id=?", (experiment.id,))
            db.executemany("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?)",
                           [(experiment.id, trial.fixture_id, trial.config_id, trial.execution_order,
                             trial.attempt, trial.model_dump_json()) for trial in experiment.results])
        jsonl_dir = self.path.parent / "experiments"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        with (jsonl_dir / f"{experiment.id}.jsonl").open("w") as stream:
            for trial in experiment.results:
                stream.write(trial.model_dump_json() + "\n")

    def create(self, experiment: Experiment) -> None:
        payload = experiment.model_copy(update={"results": [], "pairings": []}).model_dump_json()
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO experiments VALUES (?, ?, ?)", (experiment.id, experiment.created_at, payload))
        jsonl_dir = self.path.parent / "experiments"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        (jsonl_dir / f"{experiment.id}.jsonl").write_text("")

    def checkpoint(self, experiment: Experiment) -> None:
        payload = experiment.model_copy(update={"results": [], "pairings": []}).model_dump_json()
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE experiments SET payload=? WHERE id=?", (payload, experiment.id))

    def append_trial(self, experiment_id: str, trial: Any) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO trials VALUES (?, ?, ?, ?, ?, ?)",
                       (experiment_id, trial.fixture_id, trial.config_id, trial.execution_order,
                        trial.attempt, trial.model_dump_json()))
        jsonl_dir = self.path.parent / "experiments"
        jsonl_dir.mkdir(parents=True, exist_ok=True)
        with (jsonl_dir / f"{experiment_id}.jsonl").open("a") as stream:
            stream.write(trial.model_dump_json() + "\n")

    def list(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.path) as db:
            return [{"id": row[0], "created_at": row[1], "status": json.loads(row[2])["status"]}
                    for row in db.execute("SELECT id, created_at, payload FROM experiments ORDER BY created_at DESC")]

    def get(self, experiment_id: str) -> Experiment | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            trials = db.execute("SELECT payload FROM trials WHERE experiment_id=? ORDER BY execution_order",
                                (experiment_id,)).fetchall() if row else []
        if not row:
            return None
        experiment = Experiment.model_validate_json(row[0])
        experiment.results = [InferenceResult.model_validate_json(x[0]) for x in trials] if trials else []
        return experiment


def export_json(experiment: Experiment, path: Path) -> None:
    path.write_text(experiment.model_dump_json(indent=2) + "\n")


def export_csv(experiment: Experiment, path: Path) -> None:
    fields = ["fixture_id", "suite", "context", "fact_position", "config_id", "status", "answer",
              "score", "error", "prompt_token_count", "end_to_end_ms", "peak_rss_bytes", "execution_order", "attempt"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        suites = {f.id: f.suite for f in experiment.fixtures}
        fixtures = {f.id: f for f in experiment.fixtures}
        for result in experiment.results:
            fixture = fixtures.get(result.fixture_id)
            writer.writerow({"fixture_id": result.fixture_id, "suite": suites.get(result.fixture_id, ""),
                             "context": fixture.target_context_length if fixture else "",
                             "fact_position": fixture.fact_position if fixture else "",
                             "config_id": result.config_id, "status": result.status, "answer": result.answer,
                             "score": result.outcome.status if result.outcome else "", "error": result.error or "",
                             "prompt_token_count": len(result.prompt_tokens) or "",
                             "end_to_end_ms": result.end_to_end_ms or "",
                             "peak_rss_bytes": result.peak_process_rss_bytes or "",
                             "execution_order": result.execution_order, "attempt": result.attempt})


def markdown_report(experiment: Experiment) -> str:
    metrics = summarize(experiment)
    lines = ["# KV Cache Quality Benchmark experiment", "", f"- ID: `{experiment.id}`", f"- Model SHA-256: `{experiment.model_sha256}`",
             f"- Runtime SHA-256: `{experiment.runtime_sha256 or 'unavailable'}`",
             f"- Manifest SHA-256: `{experiment.manifest_sha256 or 'unavailable'}`",
             f"- llama.cpp: {experiment.runtime_version}", f"- Hardware: {experiment.hardware}",
             f"- Status: {experiment.status}", f"- Effective settings: `{experiment.settings}`", "",
             "## Results", "", "| Config | Accuracy | Delta (pp) | Retention | Regressions | Improvements | Disagreements | KV allocation by context | Peak RSS | Mean latency | Prompt tok/s | Decode tok/s |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for config in experiment.configurations:
        metric = metrics["configurations"][config.id]
        pct = f"{metric['correct']}/{metric['accuracy_denominator']} ({metric['accuracy']:.1%})" if metric["accuracy"] is not None else "unavailable"
        retention = f"{metric['retention_numerator']}/{metric['retention_denominator']} ({metric['retention']:.1%})" if metric["retention"] is not None else "insufficient baseline successes"
        latency = f"{metric['mean_end_to_end_ms']:.1f} ms" if metric["mean_end_to_end_ms"] is not None else "unavailable"
        rss = f"{metric['peak_rss_bytes'] / (1024**3):.2f} GiB" if metric["peak_rss_bytes"] else "unavailable"
        delta = metric["accuracy_delta_percentage_points"]
        delta_text = f"{delta:+.1f}" if delta is not None else "unavailable"
        ci = metric["paired_accuracy_delta_ci_95_pp"]
        if ci:
            delta_text += f" (95% paired CI {ci[0]:+.1f} to {ci[1]:+.1f})"
        allocations = metric["kv_allocation"]
        kv_values = [f"{key.split(':', 1)[0]}: {value['total_bytes'] / (1024**3):.3f} GiB"
                     for key, value in allocations.items()
                     if key.endswith(f":{config.id}") and value.get("available") and value.get("total_bytes") is not None]
        kv_text = "; ".join(kv_values) or "unavailable"
        prompt_speed = f"{metric['mean_prompt_tokens_per_second']:.1f}" if metric["mean_prompt_tokens_per_second"] is not None else "unavailable"
        decode_speed = f"{metric['mean_generation_tokens_per_second']:.1f}" if metric["mean_generation_tokens_per_second"] is not None else "unavailable"
        lines.append(f"| {config.id} ({config.cache_type_k}/{config.cache_type_v}) | {pct} | {delta_text} | {retention} | {metric['regressions']} | {metric['improvements']} | {metric['disagreements']}/{metric['paired']} | {kv_text} | {rss} | {latency} | {prompt_speed} | {decode_speed} |")
    baseline_id = metrics.get("baseline_config_id", "baseline")
    baseline = metrics["configurations"].get(baseline_id, {}).get("accuracy")
    lines += ["", "## Context accuracy", "", "| Context | Configuration | Correct | Accuracy |", "|---:|---|---:|---:|"]
    for context, group in metrics["contexts"].items():
        for config_id, item in group.items():
            value = f"{item['correct']}/{item['n']} ({item['accuracy']:.1%})" if item["accuracy"] is not None else "unavailable"
            lines.append(f"| {context} | {config_id} | {item['correct']}/{item['n']} | {value} |")
    lines += ["", f"Baseline accuracy: {baseline:.1%}" if baseline is not None else "Baseline accuracy: unavailable",
              "", "## Unsupported configurations and skipped fixtures", ""]
    lines.extend(f"- {item.get('config_id', 'configuration')}: {item.get('reason', item.get('status', 'unavailable'))}" for item in experiment.unsupported)
    if not experiment.unsupported:
        lines.append("- None recorded")
    lines += ["", "## Representative paired outcomes", ""]
    regressions = [p for p in experiment.pairings if p["classification"] == "regression"][:10]
    lines.extend(f"- `{p['fixture_id']}` {p['config_id']}: expected baseline `{p['baseline_answer']}`, candidate `{p['candidate_answer']}`" for p in regressions)
    if not regressions:
        lines.append("- No paired regressions recorded")
    lines += ["", "## Measurement limits", "", "Peak process RSS is sampled and is not KV allocation. KV allocation is reported only when explicit runtime log lines were recognized. Prompt/decode timings depend on the llama.cpp response schema. Small samples are exploratory; deterministic decoding does not guarantee bitwise-identical outputs across backends.",
              "", "Reproduce with the GGUF SHA-256 and llama-server build recorded above, using the resolved settings and fixture documents in the exported JSON. Results describe this exact model artifact, runtime, hardware, and fixture sample."]
    return "\n".join(lines) + "\n"
