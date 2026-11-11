"""Fixture/attempt/context-aligned baseline comparisons."""
from __future__ import annotations

from typing import Any

from .models import Fixture, InferenceResult
from .scoring import classify_pair, normalize


def pair_results(results: list[InferenceResult], fixtures: list[Fixture], baseline_id: str | None,
                 config_ids: list[str] | None = None, repetitions: int | None = None) -> list[dict[str, Any]]:
    config_ids = config_ids or sorted({row.config_id for row in results})
    repetitions = repetitions or max((row.attempt for row in results), default=1)
    fixture_context = {fixture.id: fixture.target_context_length for fixture in fixtures}
    by_key = {(row.fixture_id, row.attempt, fixture_context.get(row.fixture_id)): row
              for row in results if row.config_id == baseline_id}
    rows_by_key = {(row.fixture_id, row.config_id, row.attempt): row for row in results}
    pairings: list[dict[str, Any]] = []
    for config_id in config_ids:
        if config_id == baseline_id:
            continue
        for fixture in fixtures:
            context = fixture_context.get(fixture.id)
            for attempt in range(1, repetitions + 1):
                base = by_key.get((fixture.id, attempt, context))
                row = rows_by_key.get((fixture.id, config_id, attempt))
                available = bool(base and row and base.status == "complete" and row.status == "complete" and base.outcome and row.outcome)
                category = classify_pair(base.outcome if available and base else None,
                                         row.outcome if available and row else None)
                pairings.append({"fixture_id": fixture.id, "context": context, "attempt": attempt,
                                 "config_id": config_id, "classification": category,
                                 "output_disagreement": bool(available and base and row and normalize(base.answer) != normalize(row.answer)),
                                 "baseline_answer": base.answer if base else None, "candidate_answer": row.answer if row else None,
                                 "baseline_status": base.outcome.status if base and base.outcome else None,
                                 "candidate_status": row.outcome.status if row and row.outcome else None,
                                 "baseline_runtime_status": base.status if base else None,
                                 "candidate_runtime_status": row.status if row else None,
                                 "candidate_error": row.error if row else "No candidate trial recorded"})
    return pairings
