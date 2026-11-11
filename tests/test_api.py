import asyncio
from pathlib import Path

import httpx

from kv_cache_quality_benchmark import api
from kv_cache_quality_benchmark.models import Experiment, Fixture, InferenceResult, KVConfig, Outcome
from kv_cache_quality_benchmark.storage import Store


def test_dashboard_api_lists_summaries_and_filters(tmp_path: Path, monkeypatch):
    database = tmp_path / "api.sqlite"
    store = Store(database)
    fixture = Fixture(id="needle_1", suite="needle", seed=3, document="Incident X: code Z.",
                      question="What code?", expected_answer="Z", fact_spans=["code Z."],
                      target_context_length=2048, actual_token_length=32, fact_position="50%")
    experiment = Experiment(id="api-run", model_path="model.gguf", model_sha256="b" * 64,
                            runtime_path="llama-server", runtime_version="build",
                            configurations=[KVConfig(id="baseline", key="f16", value="f16")],
                            fixtures=[fixture], results=[InferenceResult(
                                fixture_id=fixture.id, config_id="baseline", status="complete", answer="Z",
                                outcome=Outcome(status="correct", correct=True))])
    store.save(experiment)
    monkeypatch.setattr(api, "store", lambda: store)
    async def check_api() -> None:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/experiments")).json()[0]["id"] == "api-run"
            assert (await client.get("/api/experiments/api-run")).status_code == 200
            summary = await client.get("/api/experiments/api-run/summary")
            assert summary.json()["metrics"]["configurations"]["baseline"]["accuracy"] == 1
            results = await client.get("/api/experiments/api-run/results?workload=needle")
            assert len(results.json()) == 1
            assert (await client.get("/api/experiments/missing")).status_code == 404

    asyncio.run(check_api())
