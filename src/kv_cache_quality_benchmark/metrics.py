"""Paired summaries with explicit denominators and no composite fidelity score."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .models import Experiment


def summarize(experiment: Experiment) -> dict[str, Any]:
    fixture_by_id = {f.id: f for f in experiment.fixtures}
    configs: dict[str, Any] = {}
    for config in experiment.configurations:
        rows = [r for r in experiment.results if r.config_id == config.id]
        valid = [r for r in rows if r.status == "complete" and r.outcome and r.outcome.correct is not None]
        correct = sum(bool(r.outcome and r.outcome.correct) for r in valid)
        pair_rows = [p for p in experiment.pairings if p["config_id"] == config.id]
        comparable = [p for p in pair_rows if p["classification"] != "comparison_unavailable"]
        baseline_successes = [p for p in comparable if p.get("baseline_status") == "correct"]
        retained = sum(p.get("candidate_status") == "correct" for p in baseline_successes)
        answer_times = [r.end_to_end_ms for r in rows if r.end_to_end_ms is not None]
        prompt_speeds = [r.timings["prompt_tokens_per_second"] for r in rows
                         if isinstance(r.timings.get("prompt_tokens_per_second"), (int, float))]
        generation_speeds = [r.timings["generation_tokens_per_second"] for r in rows
                             if isinstance(r.timings.get("generation_tokens_per_second"), (int, float))]
        rss_values = [r.peak_process_rss_bytes for r in rows if r.peak_process_rss_bytes is not None]
        unanswerable = [r for r in valid if fixture_by_id.get(r.fixture_id) and fixture_by_id[r.fixture_id].answer_type == "unanswerable"]
        fact_rows = [r for r in valid if fixture_by_id.get(r.fixture_id) and
                     fixture_by_id[r.fixture_id].suite in {"retention", "multi_fact"}]
        fact_correct = sum(bool(r.outcome and r.outcome.correct) for r in fact_rows)
        configs[config.id] = {
            "cases": len(valid), "correct": correct, "accuracy": correct / len(valid) if valid else None,
            "accuracy_denominator": len(valid),
            "retention": retained / len(baseline_successes) if baseline_successes else None,
            "retention_numerator": retained, "retention_denominator": len(baseline_successes),
            "regressions": sum(p["classification"] == "regression" for p in pair_rows),
            "improvements": sum(p["classification"] == "improvement" for p in pair_rows),
            "disagreements": sum(bool(p["output_disagreement"]) for p in pair_rows),
            "paired": len(comparable),
            "disagreement_rate": sum(bool(p["output_disagreement"]) for p in comparable) / len(comparable) if comparable else None,
            "hallucinations": sum(
                r.outcome is not None and r.outcome.status == "unsupported_assertion"
                for r in unanswerable
            ),
            "unanswerable_count": len(unanswerable),
            "hallucination_rate": (
                sum(
                    r.outcome is not None and r.outcome.status == "unsupported_assertion"
                    for r in unanswerable
                )
                / len(unanswerable)
                if unanswerable
                else None
            ),
            "critical_fact_correct": fact_correct, "critical_fact_count": len(fact_rows),
            "mean_end_to_end_ms": sum(answer_times) / len(answer_times) if answer_times else None,
            "mean_prompt_tokens_per_second": sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else None,
            "mean_generation_tokens_per_second": sum(generation_speeds) / len(generation_speeds) if generation_speeds else None,
            "peak_rss_bytes": max(rss_values) if rss_values else None,
            "kv_allocation": experiment.settings.get("kv_allocation_by_config_context", {}),
        }
    per_context: dict[str, dict[str, Any]] = {}
    for context in sorted({f.target_context_length for f in experiment.fixtures if f.target_context_length}):
        ids = {f.id for f in experiment.fixtures if f.target_context_length == context}
        per_context[str(context)] = {}
        for config in experiment.configurations:
            rows = [r for r in experiment.results if r.config_id == config.id and r.fixture_id in ids and r.outcome and r.outcome.correct is not None]
            per_context[str(context)][config.id] = {"correct": sum(bool(r.outcome and r.outcome.correct) for r in rows),
                                                   "n": len(rows), "accuracy": sum(bool(r.outcome and r.outcome.correct) for r in rows) / len(rows) if rows else None}
    baseline_id = experiment.settings.get("baseline_config_id", "baseline")
    baseline_accuracy = configs.get(baseline_id, {}).get("accuracy")
    for data in configs.values():
        data["accuracy_delta_percentage_points"] = ((data["accuracy"] - baseline_accuracy) * 100
                                                      if data["accuracy"] is not None and baseline_accuracy is not None else None)
    if baseline_id in configs:
        baseline = configs[baseline_id]
        baseline["retention"] = 1.0 if baseline["correct"] else None
        baseline["retention_numerator"] = baseline["correct"]
        baseline["retention_denominator"] = baseline["correct"]
    # Paired bootstrap over fixture IDs, averaging repetitions within each fixture first.
    for config_id, data in configs.items():
        by_fixture: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for pair in experiment.pairings:
            if pair["config_id"] != config_id or pair["classification"] == "comparison_unavailable":
                continue
            by_fixture[pair["fixture_id"]].append((float(pair["baseline_status"] == "correct"),
                                                    float(pair["candidate_status"] == "correct")))
        deltas = [sum(candidate - baseline for baseline, candidate in values) / len(values)
                  for values in by_fixture.values()]
        if len(deltas) >= 20:
            rng = random.Random(42)
            samples = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) * 100 for _ in range(2000))
            data["paired_accuracy_delta_ci_95_pp"] = [samples[49], samples[1949]]
            data["paired_bootstrap_fixtures"] = len(deltas)
        else:
            data["paired_accuracy_delta_ci_95_pp"] = None
            data["paired_bootstrap_fixtures"] = len(deltas)
    return {"baseline_config_id": baseline_id, "configurations": configs, "contexts": per_context,
            "statistical_note": "Insufficient samples for a meaningful confidence interval"}


def heatmap(experiment: Experiment) -> dict[str, dict[str, dict[str, int | None]]]:
    buckets = ("Beginning", "25%", "Middle", "75%", "End")
    output: dict[str, dict[str, dict[str, int | None]]] = defaultdict(dict)
    by_id = {fixture.id: fixture for fixture in experiment.fixtures}
    for config in experiment.configurations:
        for bucket in buckets:
            rows = []
            for row in experiment.results:
                fixture = by_id.get(row.fixture_id)
                if row.config_id != config.id or not row.outcome or row.outcome.correct is None or not fixture:
                    continue
                if fixture.fact_position and fixture.fact_position.endswith("%"):
                    index = min(4, max(0, int(float(fixture.fact_position[:-1]) / 25 + .5)))
                    category = buckets[index]
                else:
                    continue
                if category == bucket:
                    rows.append(row)
            output[config.id][bucket] = {"correct": sum(bool(r.outcome and r.outcome.correct) for r in rows), "n": len(rows)}
    return dict(output)
