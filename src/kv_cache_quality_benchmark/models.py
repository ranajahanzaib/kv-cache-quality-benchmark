"""Validated records shared across KV Cache Quality Benchmark components."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class KVConfig(BaseModel):
    id: str
    cache_type_k: str = Field(validation_alias=AliasChoices("cache_type_k", "key"))
    cache_type_v: str = Field(validation_alias=AliasChoices("cache_type_v", "value"))


DEFAULT_CONFIGS = [
    KVConfig(id="baseline", cache_type_k="f16", cache_type_v="f16"),
    KVConfig(id="q8", cache_type_k="q8_0", cache_type_v="q8_0"),
    KVConfig(id="q4", cache_type_k="q4_0", cache_type_v="q4_0"),
    KVConfig(id="mixed", cache_type_k="q8_0", cache_type_v="q4_0"),
]


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    suite: str
    seed: int
    document: str
    question: str
    expected_answer: str
    answer_type: Literal["exact", "multi", "unanswerable"] = "exact"
    fact_position: str | None = None
    fact_spans: list[str] = Field(default_factory=list)
    distractors: int = 0
    facts: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    target_context_length: int | None = None
    actual_token_length: int | None = None
    invalid_reason: str | None = None


class Outcome(BaseModel):
    status: Literal["correct", "incorrect", "abstained_correctly", "unsupported_assertion", "ambiguous", "format_error", "generation_failure", "not_scored"]
    correct: bool | None
    extracted: list[str] = Field(default_factory=list)
    fact_results: dict[str, bool] = Field(default_factory=dict)
    complete_correct: bool | None = None
    scorer_version: str = "1"


class InferenceResult(BaseModel):
    fixture_id: str
    config_id: str
    status: str
    answer: str = ""
    outcome: Outcome | None = None
    error: str | None = None
    prompt_tokens: list[int] = Field(default_factory=list)
    generation_tokens: list[int] = Field(default_factory=list)
    timings: dict[str, Any] = Field(default_factory=dict)
    memory_rss_bytes: int | None = None
    started_at: str = ""
    completed_at: str = ""
    execution_order: int = 0
    attempt: int = 1
    end_to_end_ms: float | None = None
    process_rss_before_bytes: int | None = None
    peak_process_rss_bytes: int | None = None


class Experiment(BaseModel):
    id: str
    name: str = "local"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "complete"
    model_path: str
    model_sha256: str
    runtime_path: str
    runtime_version: str
    runtime_sha256: str = ""
    manifest_sha256: str = ""
    hardware: dict[str, Any] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    fixtures: list[Fixture] = Field(default_factory=list)
    configurations: list[KVConfig] = Field(default_factory=list)
    results: list[InferenceResult] = Field(default_factory=list)
    unsupported: list[dict[str, Any]] = Field(default_factory=list)
    pairings: list[dict[str, Any]] = Field(default_factory=list)
