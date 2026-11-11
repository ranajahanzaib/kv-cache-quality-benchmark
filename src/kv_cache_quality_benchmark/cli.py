"""Command line interface."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path
from typing import Any, Literal

import typer

from .config import load_config
from .models import Experiment, InferenceResult
from .runner import run_experiment
from .runtime import LlamaServer, RuntimeErrorDetail, inspect_runtime, sha256_file
from .storage import DEFAULT_DB, Store, export_csv, export_json, markdown_report

app = typer.Typer(no_args_is_help=True, help="Measure behavioral changes across llama.cpp KV-cache configurations.")


@app.command()
def doctor(binary: str = typer.Option("llama-server", help="Path or executable name"),
           model: Path | None = typer.Option(None, help="Optional GGUF to validate")) -> None:
    """Check local runtime availability and capabilities."""
    runtime_error = None
    try:
        info = inspect_runtime(binary)
    except RuntimeErrorDetail as exc:
        info, runtime_error = None, str(exc)
    packages = {name: importlib.util.find_spec(name) is not None for name in ("pydantic", "typer", "httpx", "psutil", "fastapi")}
    report = {"os": platform.platform(), "architecture": platform.machine(),
              "system_memory_available_bytes": __import__("psutil").virtual_memory().available,
              "runtime": {k: v for k, v in info.items() if k != "help"} if info else {"available": False, "reason": runtime_error},
              "python_packages": packages,
              "cache_formats": ["f16", "q8_0", "q4_0"],
              "dashboard_dependency_available": packages["fastapi"] and packages["httpx"]}
    if model:
        report["model"] = {"path": str(model.resolve()), "readable": model.is_file(),
                           "sha256": sha256_file(model) if model.is_file() else None}
    typer.echo(json.dumps(report, indent=2))
    invalid_model = bool(model and not model.is_file())
    if not info or invalid_model or not all(packages.values()):
        raise typer.Exit(2)


@app.command()
def probe(model: Path = typer.Option(..., exists=True, readable=True),
          binary: str = typer.Option("llama-server"), context: int = typer.Option(2048, min=256),
          gpu_layers: int = typer.Option(99)) -> None:
    """Try each built-in cache format against a local GGUF."""
    from .models import DEFAULT_CONFIGS
    report: dict[str, Any] = {"model": str(model.resolve()), "sha256": sha256_file(model), "context": context, "configurations": []}
    for cfg in DEFAULT_CONFIGS:
        try:
            with LlamaServer(binary, model, context=context, cache_k=cfg.cache_type_k,
                             cache_v=cfg.cache_type_v, gpu_layers=gpu_layers) as server:
                server.complete(server.prepare_prompt("", "Reply with only READY."), 2)
                logs = server.read_logs()
                verified = cfg.cache_type_k in logs and cfg.cache_type_v in logs
                report["configurations"].append({"id": cfg.id,
                                                  "status": "supported" if verified else "unverified",
                                                  "reason": None if verified else "Inference completed, but runtime logs did not confirm effective cache precision",
                                                  "version": server.runtime["version"]})
        except Exception as exc:  # noqa: BLE001 - probe reports failures per configuration
            report["configurations"].append({"id": cfg.id, "status": "runtime_failed",
                                              "reason": f"{type(exc).__name__}: {exc}"})
    typer.echo(json.dumps(report, indent=2))


@app.command()
def generate(model: Path = typer.Option(..., exists=True, readable=True),
             binary: str = typer.Option("llama-server"), preset: str = typer.Option("smoke"),
             seed: int = typer.Option(42), context: int = typer.Option(2048, min=256),
             output: Path = typer.Option(Path("fixtures.jsonl")), gpu_layers: int = typer.Option(99)) -> None:
    """Generate model-tokenized fixtures and report their validation status."""
    from .fixtures import generate_fixtures, resize_fixtures
    from .runner import PRESETS
    from .runtime import LlamaServer
    if preset not in PRESETS:
        raise typer.BadParameter("preset must be smoke, standard, or extended")
    try:
        with LlamaServer(binary, model, context=context, cache_k="f16", cache_v="f16", gpu_layers=gpu_layers) as server:
            fixtures = resize_fixtures(generate_fixtures(seed, PRESETS[preset]["per_suite"]), server.tokenize,
                                       context=context, output_budget=PRESETS[preset]["tokens"],
                                       prompt_builder=server.prepare_prompt,
                                       token_position=server.token_position)
    except Exception as exc:
        typer.echo(f"Fixture generation failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f.model_dump_json() + "\n" for f in fixtures))
    typer.echo(json.dumps([{"id": f.id, "tokens": f.actual_token_length, "fact_position": f.fact_position,
                           "valid": f.invalid_reason is None, "reason": f.invalid_reason} for f in fixtures], indent=2))
    typer.echo(f"Saved {len(fixtures)} fixtures to {output}")


@app.command("run")
def run(model: Path | None = typer.Option(None, exists=True, readable=True),
        binary: str = typer.Option("llama-server"), preset: str = typer.Option("smoke"),
        context: int | None = typer.Option(None, min=256), cache_config: list[str] = typer.Option([]),
        per_suite: int | None = typer.Option(None, min=1), gpu_layers: int = typer.Option(99),
        timeout: float = typer.Option(120), database: Path = typer.Option(DEFAULT_DB),
        config_file: Path | None = typer.Option(None, "--config"), repetitions: int | None = typer.Option(None, min=1),
        seed: int | None = typer.Option(None)) -> None:
    """Run and save an experiment using a local GGUF and llama-server."""
    if preset not in {"smoke", "standard", "extended"}:
        raise typer.BadParameter("preset must be smoke, standard, or extended")
    store = Store(database)
    state: dict[str, Any] = {"created": False, "experiment": None}

    def persist_progress(experiment: Experiment, trial: InferenceResult | None) -> None:
        state["experiment"] = experiment
        if not state["created"]:
            store.create(experiment)
            state["created"] = True
        elif experiment.status != "running":
            store.save(experiment)
        elif trial is not None:
            store.append_trial(experiment.id, trial)
            total = len(experiment.results)
            if total == 1 or total % 10 == 0:
                fixture = next((item for item in experiment.fixtures if item.id == trial.fixture_id), None)
                typer.echo(f"Completed {total} trials · context {fixture.target_context_length if fixture else '?'} · {trial.config_id} · {fixture.suite if fixture else trial.fixture_id}", err=True)
        else:
            store.checkpoint(experiment)
    try:
        resolved = load_config(config_file) if config_file else None
        if resolved:
            model = model or resolved.runtime.model
            binary = binary if binary != "llama-server" else resolved.runtime.binary
            preset = preset if preset != "smoke" else resolved.experiment.preset
            gpu_layers = gpu_layers if gpu_layers != 99 else (99 if resolved.runtime.gpu_layers == "all" else resolved.runtime.gpu_layers)
        if model is None:
            raise ValueError("provide --model or set runtime.model in --config")
        selected_configs = cache_config or ([c.id for c in resolved.cache_configs] if resolved and resolved.cache_configs else None)
        contexts = ([context] if context is not None else resolved.contexts or None) if resolved else None
        experiment = run_experiment(model, binary, preset, context=context, configs=selected_configs,
                                    custom_configs=resolved.cache_configs if resolved and resolved.cache_configs else None,
                                    contexts=contexts, per_suite=per_suite, gpu_layers=int(gpu_layers), timeout=timeout,
                                    repetitions=repetitions or (resolved.experiment.repetitions if resolved else (3 if preset == "extended" else 1)),
                                    seed=seed if seed is not None else (resolved.experiment.seed if resolved else 42),
                                    sample_interval_ms=resolved.measurement.sample_interval_ms if resolved else 100,
                                    output_tokens=resolved.generation.max_output_tokens if resolved else None,
                                    workloads=resolved.workloads if resolved else None,
                                    name=resolved.experiment.name if resolved else "local",
                                    flash_attention=resolved.runtime.flash_attention if resolved else "on",
                                    progress=persist_progress,
                                    generation_seed=resolved.generation.seed if resolved and seed is None else seed,
                                    temperature=resolved.generation.temperature if resolved else 0,
                                    process_memory=resolved.measurement.process_memory if resolved else True)
        if resolved:
            experiment.settings["resolved_config"] = resolved.model_dump(mode="json")
    except KeyboardInterrupt:
        partial = state.get("experiment")
        if partial is not None:
            partial.status = "interrupted"
            store.save(partial)
        typer.echo("Experiment interrupted; completed trials were saved.", err=True)
        raise typer.Exit(130)
    except Exception as exc:
        partial = state.get("experiment")
        if partial is not None and partial.status == "running":
            partial.status = "failed"
            store.save(partial)
        typer.echo(f"Experiment failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(2) from exc
    store.save(experiment)
    typer.echo(markdown_report(experiment))
    typer.echo(f"Saved experiment {experiment.id} to {database}")
    if experiment.status == "failed":
        raise typer.Exit(1)


@app.command()
def report(experiment_id: str, database: Path = typer.Option(DEFAULT_DB)) -> None:
    """Print a saved experiment's paired metrics and diagnostics."""
    experiment = Store(database).get(experiment_id)
    if experiment is None:
        typer.echo("Experiment not found", err=True)
        raise typer.Exit(2)
    typer.echo(markdown_report(experiment))


@app.command()
def export(experiment_id: str, output: Path = typer.Option(Path("kv-cache-quality-benchmark-export")),
           format: Literal["all", "json", "csv", "markdown"] = typer.Option("all"),
           database: Path = typer.Option(DEFAULT_DB)) -> None:
    """Export a saved experiment as JSON, CSV, and Markdown."""
    experiment = Store(database).get(experiment_id)
    if experiment is None:
        typer.echo("Experiment not found", err=True)
        raise typer.Exit(2)
    if format == "all":
        output.mkdir(parents=True, exist_ok=True)
        export_json(experiment, output / "experiment.json")
        export_csv(experiment, output / "results.csv")
        (output / "report.md").write_text(markdown_report(experiment))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            export_json(experiment, output)
        elif format == "csv":
            export_csv(experiment, output)
        else:
            output.write_text(markdown_report(experiment))
    typer.echo(f"Exported to {output}")


@app.command()
def dashboard(host: str = typer.Option(
                  "127.0.0.1", help="Bind address; the API has no authentication, so use loopback only."),
              port: int = typer.Option(8000),
              database: Path = typer.Option(DEFAULT_DB)) -> None:
    """Serve saved experiment data from the local API."""
    import uvicorn
    os.environ["CACHEFIDELITY_DB"] = str(database.expanduser())
    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if frontend_dist.is_dir():
        typer.echo(f"Dashboard available at http://{host}:{port}/ (no browser opened)")
    else:
        typer.echo(f"API available at http://{host}:{port}/api. For the UI run `cd frontend && npm run dev` in another terminal.")
    uvicorn.run("kv_cache_quality_benchmark.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
