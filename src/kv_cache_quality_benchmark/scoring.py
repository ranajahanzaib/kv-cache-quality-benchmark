"""Conservative deterministic scoring for short, controlled answers."""
from __future__ import annotations

import re

from .models import Fixture, Outcome

_CLEAN = re.compile(r"^[\s\"'`*_]+|[\s\"'`*_.,;:!?]+$")


def normalize(value: str) -> str:
    return _CLEAN.sub("", value.strip()).casefold()


def score(fixture: Fixture, answer: str) -> Outcome:
    raw = answer.strip()
    if not raw:
        return Outcome(status="format_error", correct=False)
    if fixture.answer_type == "unanswerable":
        if normalize(raw) in {"not_found", "not found", "unknown", "not provided"}:
            return Outcome(status="abstained_correctly", correct=True, extracted=["NOT_FOUND"])
        return Outcome(status="unsupported_assertion", correct=False, extracted=[raw])
    expected = normalize(fixture.expected_answer)
    # A short standalone answer is required; contradictory prose is not accepted.
    candidates = [normalize(line) for line in raw.splitlines() if normalize(line)]
    if len(candidates) != 1:
        return Outcome(status="ambiguous", correct=False, extracted=candidates)
    actual = candidates[0]
    if fixture.answer_type == "multi" and fixture.required_facts:
        values = [normalize(part) for part in raw.split("|")]
        fact_results = {fact: (i < len(values) and values[i] == normalize(fact))
                        for i, fact in enumerate(fixture.required_facts)}
        whole = len(values) == len(fixture.required_facts) and all(fact_results.values())
        return Outcome(status="correct" if whole else "incorrect", correct=whole,
                       extracted=[raw], fact_results=fact_results, complete_correct=whole)
    if actual == expected:
        return Outcome(status="correct", correct=True, extracted=[raw], complete_correct=True)
    if expected in actual or actual in expected:
        return Outcome(status="format_error", correct=False, extracted=[raw])
    return Outcome(status="incorrect", correct=False, extracted=[raw])


def classify_pair(baseline: Outcome | None, candidate: Outcome | None) -> str:
    if baseline is None or candidate is None or baseline.correct is None or candidate.correct is None:
        return "comparison_unavailable"
    if baseline.correct and candidate.correct:
        return "preserved_success"
    if baseline.correct and not candidate.correct:
        return "regression"
    if not baseline.correct and candidate.correct:
        return "improvement"
    return "persistent_failure"
