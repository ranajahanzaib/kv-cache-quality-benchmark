from pathlib import Path

from kv_cache_quality_benchmark.config import load_config
from kv_cache_quality_benchmark.fixtures import generate_fixtures, resize_fixtures
from kv_cache_quality_benchmark.measurement import parse_context_metadata, parse_kv_allocation, response_timings
from kv_cache_quality_benchmark.metrics import heatmap, summarize
from kv_cache_quality_benchmark.models import Experiment, Fixture, InferenceResult, KVConfig, Outcome
from kv_cache_quality_benchmark.pairing import pair_results
from kv_cache_quality_benchmark.scoring import classify_pair, score
from kv_cache_quality_benchmark.storage import Store, export_csv, export_json, markdown_report


def whitespace_tokenizer(text: str) -> list[int]:
    return list(range(len(text.split())))


def test_generation_is_seeded_and_ids_unique_and_targets_not_repeated():
    first = generate_fixtures(5, 5)
    assert first == generate_fixtures(5, 5)
    assert len({f.id for f in first}) == len(first)
    for fixture in first:
        if fixture.suite == "needle":
            target = fixture.expected_answer
            assert fixture.document.count(target) == 1


def test_fixture_resizing_uses_tokenizer_budget_and_fact_location():
    source = next(f for f in generate_fixtures(5, 5) if f.suite == "needle" and f.fact_position == "75%")
    resized = resize_fixtures([source], whitespace_tokenizer, context=2048, output_budget=64)[0]
    assert resized.actual_token_length == len(whitespace_tokenizer(
        "Answer using only the supplied document. If the information is absent, answer NOT_FOUND. "
        "Return only the answer value.\n\nDOCUMENT:\n" + resized.document +
        "\n\nQUESTION:\n" + resized.question + "\n\nANSWER:"))
    assert resized.actual_token_length + 64 <= 2048
    assert resized.fact_position is not None
    assert 60 <= float(resized.fact_position.rstrip("%")) <= 90


def test_exact_multi_and_unanswerable_scoring():
    exact = Fixture(id="x", suite="needle", seed=1, document="", question="", expected_answer="VELVET-482")
    assert score(exact, " VELVET-482\n").status == "correct"
    assert score(exact, "VELVET-428").status == "incorrect"
    assert score(exact, "VELVET-482\nVELVET-428").status == "ambiguous"
    assert score(exact, "").status == "format_error"
    assert score(exact, "The answer is VELVET-482, but maybe VELVET-428").status == "format_error"
    multi = Fixture(id="m", suite="multi_fact", seed=1, document="", question="", expected_answer="A|B",
                    answer_type="multi", required_facts=["A", "B"])
    partial = score(multi, "A|C")
    assert partial.correct is False and partial.fact_results == {"A": True, "B": False}
    assert score(multi, "A|B").complete_correct is True
    absent = exact.model_copy(update={"answer_type": "unanswerable", "expected_answer": "NOT_FOUND"})
    assert score(absent, "NOT_FOUND").status == "abstained_correctly"
    assert score(absent, "Probably VELVET-482").status == "unsupported_assertion"


def test_pair_classification_and_disagreement_are_distinct():
    yes = Outcome(status="correct", correct=True)
    no = Outcome(status="incorrect", correct=False)
    assert classify_pair(yes, yes) == "preserved_success"
    assert classify_pair(yes, no) == "regression"
    assert classify_pair(no, yes) == "improvement"
    assert classify_pair(no, no) == "persistent_failure"
    assert classify_pair(None, yes) == "comparison_unavailable"
    fixture = Fixture(id="pair", suite="needle", seed=1, document="", question="", expected_answer="TARGET",
                      target_context_length=2048)
    paired = pair_results([
        InferenceResult(fixture_id="pair", config_id="baseline", status="complete", answer="wrong A", outcome=no),
        InferenceResult(fixture_id="pair", config_id="q4", status="complete", answer="wrong B", outcome=no),
    ], [fixture], "baseline", ["baseline", "q4"], 1)[0]
    assert paired["classification"] == "persistent_failure"
    assert paired["output_disagreement"] is True
    missing = pair_results([], [fixture], "baseline", ["baseline", "q4"], 1)[0]
    assert missing["classification"] == "comparison_unavailable"


def test_memory_and_timing_provenance():
    parsed = parse_kv_allocation("llama_kv_cache: KV buffer size = 128.5 MiB")
    assert parsed["available"] is True
    assert parsed["total_bytes"] == int(128.5 * 1024**2)
    assert parse_kv_allocation("no allocation diagnostic") == {
        "available": False, "allocations": [], "total_bytes": None, "provenance": "unavailable"}
    timing = response_timings({"timings": {"prompt_per_second": 20.0, "predicted_per_second": 4.0}}, 123.0)
    assert timing["end_to_end_ms"] == 123.0
    assert timing["prompt_tokens_per_second"] == 20.0
    assert parse_context_metadata("n_ctx_train = 32768\nn_ctx = 4096") == {
        "effective_context": 4096, "training_context": 32768}


def make_experiment() -> Experiment:
    fixtures = generate_fixtures(1, 1)
    needle = next(f for f in fixtures if f.suite == "needle")
    return Experiment(id="e1", model_path="m.gguf", model_sha256="a" * 64,
                      runtime_path="llama-server", runtime_version="version",
                      configurations=[KVConfig(id="baseline", key="f16", value="f16"),
                                      KVConfig(id="q4", key="q4_0", value="q4_0")],
                      fixtures=[needle.model_copy(update={"target_context_length": 2048,
                                                         "fact_position": "50.0%"})],
                      results=[InferenceResult(fixture_id=needle.id, config_id="baseline", status="complete",
                                               answer="VELVET-482", outcome=Outcome(status="correct", correct=True),
                                               peak_process_rss_bytes=2_000_000_000),
                               InferenceResult(fixture_id=needle.id, config_id="q4", status="complete",
                                               answer="VELVET-428", outcome=Outcome(status="incorrect", correct=False),
                                               peak_process_rss_bytes=1_000_000_000)],
                      pairings=[{"fixture_id": needle.id, "config_id": "q4", "classification": "regression",
                                 "output_disagreement": True, "baseline_status": "correct",
                                 "candidate_status": "incorrect", "baseline_answer": "VELVET-482",
                                 "candidate_answer": "VELVET-428", "attempt": 1, "context": 2048}])


def test_storage_export_metrics_and_report(tmp_path: Path):
    experiment = make_experiment()
    db = Store(tmp_path / "x.db")
    db.save(experiment)
    assert db.get("e1") == experiment
    assert len(db.list()) == 1
    export_json(experiment, tmp_path / "out.json")
    export_csv(experiment, tmp_path / "out.csv")
    assert '"id": "e1"' in (tmp_path / "out.json").read_text()
    assert "fixture_id" in (tmp_path / "out.csv").read_text()
    metrics = summarize(experiment)
    assert metrics["configurations"]["q4"]["regressions"] == 1
    assert metrics["configurations"]["q4"]["accuracy_delta_percentage_points"] == -100
    assert heatmap(experiment)["baseline"]["Middle"] == {"correct": 1, "n": 1}
    assert "VELVET-428" in markdown_report(experiment)


def test_incremental_trial_persistence_survives_interruption(tmp_path: Path):
    experiment = make_experiment().model_copy(update={"status": "running", "results": []})
    trial = InferenceResult(fixture_id=experiment.fixtures[0].id, config_id="baseline",
                            execution_order=1, status="complete", answer="answer")
    store = Store(tmp_path / "partial.db")
    store.create(experiment)
    store.checkpoint(experiment)
    store.append_trial(experiment.id, trial)
    reopened = Store(tmp_path / "partial.db").get(experiment.id)
    assert reopened is not None and reopened.status == "running"
    assert reopened.results == [trial]
    jsonl = tmp_path / "experiments" / f"{experiment.id}.jsonl"
    assert jsonl.read_text().count("\n") == 1


def test_config_validation_and_relative_model_path(tmp_path: Path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("experiment:\n  seed: 7\nruntime:\n  model: model.gguf\ncache_configs:\n  - {id: base, key: f16, value: f16}\n")
    config = load_config(config_path)
    assert config.experiment.seed == 7
    assert config.runtime.model == tmp_path / "model.gguf"
    config_path.write_text("contexts: [128]\n")
    try:
        load_config(config_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid context should be rejected")
