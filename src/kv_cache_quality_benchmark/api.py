"""Read-only local dashboard API over persisted experiments."""
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from .metrics import heatmap, summarize
from .storage import DEFAULT_DB, Store

app = FastAPI(title="KV Cache Quality Benchmark API", version="0.1.0")


def store() -> Store:
    return Store(Path(os.environ.get("CACHEFIDELITY_DB", str(DEFAULT_DB))))


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]:
    return store().list()


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict[str, Any]:
    result = store().get(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result.model_dump(mode="json")


@app.get("/api/experiments/{experiment_id}/summary")
def summary(experiment_id: str) -> dict[str, Any]:
    result = store().get(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"metrics": summarize(result), "fact_position": heatmap(result)}


@app.get("/api/experiments/{experiment_id}/results")
def results(experiment_id: str, config_id: str | None = None, workload: str | None = None,
            status: str | None = None, limit: int = Query(default=1000, ge=1, le=10000)) -> list[dict[str, Any]]:
    result = store().get(experiment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    suites = {fixture.id: fixture.suite for fixture in result.fixtures}
    selected = [row.model_dump(mode="json") for row in result.results
                if (config_id is None or row.config_id == config_id)
                and (workload is None or suites.get(row.fixture_id) == workload)
                and (status is None or (row.outcome.status if row.outcome else row.status) == status)]
    return selected[:limit]


_frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="dashboard")
