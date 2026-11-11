"""Controlled paired experiment execution."""
from __future__ import annotations

import platform
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psutil

from .fixtures import generate_fixtures, resize_fixtures
from .measurement import (
    ProcessMemorySampler,
    parse_context_metadata,
    parse_kv_allocation,
    response_timings,
)
from .models import DEFAULT_CONFIGS, Experiment, Fixture, InferenceResult, KVConfig, Outcome
from .pairing import pair_results
from .runtime import LlamaServer, RuntimeErrorDetail, inspect_runtime
from .scoring import score

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"contexts": [2048], "configs": ["baseline", "q8", "q4"], "per_suite": 1, "tokens": 24},
    "standard": {"contexts": [2048, 4096, 8192], "configs": ["baseline", "q8", "q4"], "per_suite": 5, "tokens": 64},
    "extended": {"contexts": [2048, 4096, 8192, 16384], "configs": ["baseline", "q8", "q4", "mixed"], "per_suite": 12, "tokens": 64},
}


def run_experiment(model: Path, binary: str, preset: str = "smoke", *, context: int | None = None,
                   configs: list[str] | None = None, per_suite: int | None = None,
                   gpu_layers: int = 99, timeout: float = 120, repetitions: int | None = None,
                   contexts: list[int] | None = None, seed: int = 42,
                   sample_interval_ms: int = 100, custom_configs: list[KVConfig] | None = None,
                   output_tokens: int | None = None, workloads: list[str] | None = None,
                   name: str = "local", flash_attention: str = "on",
                   progress: Callable[[Experiment, InferenceResult | None], None] | None = None,
                   generation_seed: int | None = None, temperature: float = 0,
                   process_memory: bool = True) -> Experiment:
    settings = PRESETS[preset]
    repetitions = repetitions or (3 if preset == "extended" else 1)
    selected = configs or ([c.id for c in custom_configs] if custom_configs else settings["configs"])
    all_configs = {c.id: c for c in (custom_configs or DEFAULT_CONFIGS)}
    unknown = set(selected) - all_configs.keys()
    if unknown:
        raise ValueError(f"Unknown cache configuration(s): {', '.join(sorted(unknown))}")
    requested_contexts = contexts or ([context] if context else settings["contexts"])
    output_budget = output_tokens or int(settings["tokens"])
    sampling_seed = seed if generation_seed is None else generation_seed
    base_fixtures = generate_fixtures(seed=seed, per_suite=per_suite or settings["per_suite"])
    if workloads:
        aliases = {"needle_retrieval": "needle", "critical_fact_retention": "retention",
                   "distractor_resistance": "distractor", "multi_fact_reasoning": "multi_fact"}
        normalized = {aliases.get(item, item) for item in workloads}
        unknown_workloads = normalized - {"needle", "retention", "distractor", "multi_fact", "unanswerable"}
        if unknown_workloads:
            raise ValueError(f"Unknown workload(s): {', '.join(sorted(unknown_workloads))}")
        base_fixtures = [fixture for fixture in base_fixtures if fixture.suite in normalized]
    runtime = inspect_runtime(binary)
    model_path = model.expanduser().resolve()
    from .runtime import sha256_file
    digest = sha256_file(model_path) if model_path.is_file() else ""
    experiment = Experiment(id=str(uuid.uuid4()), name=name, status="running", model_path=str(model_path), model_sha256=digest,
                            runtime_path=runtime["path"], runtime_version=runtime["version"],
                            runtime_sha256=sha256_file(Path(runtime["path"])),
                            hardware={"platform": platform.platform(), "machine": platform.machine(),
                                      "logical_cpu_count": psutil.cpu_count(),
                                      "system_memory_bytes": psutil.virtual_memory().total},
                            settings={"contexts": requested_contexts, "preset": preset,
                                      "temperature": temperature, "seed": sampling_seed, "gpu_layers": gpu_layers,
                                      "flash_attention": flash_attention,
                                      "runtime_backend_build_mentions": runtime["backends"],
                                      "repetitions": repetitions, "sample_interval_ms": sample_interval_ms,
                                      "generation_max_tokens": output_budget, "fixture_seed": seed,
                                      "process_memory_sampling": process_memory,
                                      "tokenizer_source": "llama-server /tokenize", "kv_allocation": "unavailable unless runtime reports it"},
                            fixtures=[], configurations=[all_configs[c] for c in selected])
    baseline_config_id = next((cfg.id for cfg in experiment.configurations
                               if cfg.cache_type_k.lower() == "f16" and cfg.cache_type_v.lower() == "f16"), None)
    experiment.settings["baseline_config_id"] = baseline_config_id
    if progress:
        progress(experiment, None)
    canonical_tokens_by_context: dict[tuple[int, str], list[int]] = {}
    fixtures_by_context: dict[int, list[Fixture]] = {}
    capability_records: dict[str, dict[str, Any]] = {}
    order = 0
    execution_schedule: list[dict[str, Any]] = []
    for context_index, ctx in enumerate(requested_contexts):
      ordered_configs = experiment.configurations[context_index % len(experiment.configurations):] + experiment.configurations[:context_index % len(experiment.configurations)]
      for config in ordered_configs:
        execution_schedule.append({"context": ctx, "config_id": config.id})
        stage = "initialization"
        try:
            with LlamaServer(binary, model_path, context=ctx,
                             cache_k=config.cache_type_k, cache_v=config.cache_type_v,
                             gpu_layers=gpu_layers, flash_attn=flash_attention, timeout=timeout) as server:
                caps = set(server.runtime["capabilities"])
                effective_constraints = {"single_slot": "--parallel" in caps,
                                         "flash_attention": "--flash-attn" in caps,
                                         "prompt_cache_disabled_cli": "--no-cache-prompt" in caps,
                                         "context_shift_disabled": "--no-context-shift" in caps,
                                         "auto_fit_disabled": "--fit" in caps,
                                         "cache_reuse_disabled": "--cache-reuse" in caps,
                                         "kv_offload_explicit": "--kv-offload" in caps}
                experiment.settings.setdefault("runtime_capabilities", server.runtime["capabilities"])
                experiment.settings.setdefault("enforced_constraints", effective_constraints)
                experiment.settings.setdefault("runtime_confounders", [name for name, enforced in effective_constraints.items() if not enforced])
                # A real non-scored completion is the minimum cache-format compatibility probe.
                stage = "runtime_probe"
                startup_logs = server.read_logs()
                context_metadata = parse_context_metadata(startup_logs)
                experiment.settings.setdefault("runtime_context_by_config", {})[f"{ctx}:{config.id}"] = context_metadata
                effective_context = context_metadata["effective_context"]
                training_context = context_metadata["training_context"]
                if effective_context is not None and effective_context != ctx:
                    reason = f"Runtime allocated context {effective_context}, requested {ctx}; this comparison is invalid"
                    experiment.unsupported.append({"config_id": config.id, "context": ctx,
                                                   "status": "settings_mismatch", "reason": reason})
                    capability_records[f"{ctx}:{config.id}"] = {"status": "unverified", "reason": reason}
                    if progress:
                        progress(experiment, None)
                    continue
                if training_context is not None and ctx > training_context:
                    reason = f"Requested context {ctx} exceeds GGUF training context {training_context}; no context extension is applied"
                    experiment.unsupported.append({"config_id": config.id, "context": ctx,
                                                   "status": "unsupported", "reason": reason})
                    capability_records[f"{ctx}:{config.id}"] = {"status": "unsupported", "reason": reason}
                    if progress:
                        progress(experiment, None)
                    continue
                server.complete(server.prepare_prompt("", "Reply with only READY."), 2,
                                seed=sampling_seed, temperature=temperature)
                startup_logs = server.read_logs()
                runtime_lines = [line.strip() for line in startup_logs.splitlines()
                                 if any(term in line.lower() for term in ("model", "n_ctx", "kv", "buffer", "backend", "metal"))]
                experiment.settings.setdefault("runtime_log_excerpt", runtime_lines[-100:])
                if server.process:
                    experiment.settings.setdefault("process_rss_after_startup", {})[f"{ctx}:{config.id}"] = psutil.Process(server.process.pid).memory_info().rss
                effective_confirmed = config.cache_type_k in startup_logs and config.cache_type_v in startup_logs
                capability_records[f"{ctx}:{config.id}"] = {
                    "status": "supported" if effective_confirmed else "unverified",
                    "reason": None if effective_confirmed else "Probe inference succeeded; runtime output did not confirm effective cache precision"}
                stage = "fixture_preparation"
                if ctx not in fixtures_by_context:
                    resized = resize_fixtures([f.model_copy(deep=True) for f in base_fixtures], server.tokenize,
                                              context=ctx, output_budget=output_budget,
                                              prompt_builder=server.prepare_prompt,
                                              token_position=server.token_position)
                    for fixture in resized:
                        fixture.id = f"{fixture.id}_c{ctx}"
                    fixtures_by_context[ctx] = resized
                    experiment.fixtures.extend(resized)
                    if progress:
                        progress(experiment, None)
                fixtures = fixtures_by_context[ctx]
                # Tokenize and validate all prompts before any scored inference for this configuration.
                prompts: dict[str, str] = {}
                token_map: dict[str, list[int]] = {}
                for fixture in fixtures:
                    if fixture.invalid_reason:
                        experiment.unsupported.append({"config_id": config.id, "fixture_id": fixture.id,
                                                       "context": ctx, "reason": fixture.invalid_reason})
                        continue
                    prompt = server.prepare_prompt(fixture.document, fixture.question)
                    tokens = server.tokenize(prompt)
                    if len(tokens) + output_budget > ctx:
                        experiment.unsupported.append({"config_id": config.id, "fixture_id": fixture.id,
                                                       "context": ctx, "reason": "Prompt plus generation budget exceeds context"})
                        continue
                    prompts[fixture.id], token_map[fixture.id] = prompt, tokens
                token_mismatches = [
                    fid
                    for fid, values in token_map.items()
                    if (ctx, fid) in canonical_tokens_by_context
                    and canonical_tokens_by_context[(ctx, fid)] != values
                ]
                if token_mismatches:
                    experiment.unsupported.append({"config_id": config.id,
                                                   "reason": f"Tokenizer token IDs differ for fixtures: {', '.join(token_mismatches)}"})
                    continue
                canonical_tokens_by_context.update({(ctx, fid): vals for fid, vals in token_map.items()})
                for fixture in fixtures:
                    if fixture.id not in token_map:
                        continue
                    for attempt in range(1, repetitions + 1):
                        start = datetime.now(UTC).isoformat()
                        tick = time.perf_counter()
                        try:
                            sampler = ProcessMemorySampler(server.process.pid, sample_interval_ms) if process_memory and server.process else None
                            if sampler:
                                sampler.__enter__()
                            try:
                                response = server.complete(prompts[fixture.id], output_budget,
                                                           seed=sampling_seed, temperature=temperature)
                            finally:
                                if sampler:
                                    sampler.__exit__(None, None, None)
                            answer = response["content"]
                            elapsed = (time.perf_counter() - tick) * 1000
                            proc = psutil.Process(server.process.pid) if server.process else None
                            after = proc.memory_info().rss if proc else None
                            timing_data = response_timings(response, elapsed)
                            timing_data.update({k: response[k] for k in ("generation_settings", "truncated") if k in response})
                            effective = response.get("generation_settings")
                            mismatches: dict[str, Any] = {}
                            if isinstance(effective, dict):
                                for key, expected in (("temperature", temperature), ("top_k", 1), ("seed", sampling_seed), ("n_predict", output_budget)):
                                    if key in effective and effective[key] != expected:
                                        mismatches[key] = {"requested": expected, "effective": effective[key]}
                            status = "complete"
                            outcome = score(fixture, answer)
                            error = None
                            if response.get("truncated") is True:
                                status, outcome, error = "context_overflow", Outcome(status="not_scored", correct=None), "Runtime marked completion input as truncated"
                            elif mismatches:
                                status, outcome, error = "settings_mismatch", Outcome(status="not_scored", correct=None), f"Effective generation settings differ: {mismatches}"
                                timing_data["settings_mismatches"] = mismatches
                            elif not answer.strip():
                                status, outcome, error = "empty_generation", Outcome(status="generation_failure", correct=None), "llama-server returned empty content"
                            order += 1
                            result = InferenceResult(fixture_id=fixture.id, config_id=config.id, status=status,
                                                     answer=answer, outcome=outcome, error=error, prompt_tokens=token_map[fixture.id],
                                                     generation_tokens=response.get("tokens", []) if isinstance(response.get("tokens"), list) else [],
                                                     timings=timing_data,
                                                     memory_rss_bytes=after, process_rss_before_bytes=sampler.initial if sampler else None,
                                                     peak_process_rss_bytes=(
                                                         max(sampler.peak or 0, after or 0)
                                                         if sampler is not None
                                                         else after
                                                     ),
                                                     end_to_end_ms=elapsed, started_at=start,
                                                     completed_at=datetime.now(UTC).isoformat(),
                                                     execution_order=order, attempt=attempt)
                        except Exception as exc:  # noqa: BLE001 - persist an explicit failed trial
                            order += 1
                            if server.process and server.process.poll() is not None:
                                error_status = "process_crash"
                            elif isinstance(exc, httpx.TimeoutException):
                                error_status = "request_timeout"
                            elif isinstance(exc, (RuntimeErrorDetail, ValueError)):
                                error_status = "invalid_server_response"
                            elif isinstance(exc, httpx.HTTPError):
                                error_status = "request_failed"
                            else:
                                error_status = "generation_failure"
                            result = InferenceResult(fixture_id=fixture.id, config_id=config.id, status=error_status,
                                                     outcome=Outcome(status="generation_failure", correct=None),
                                                     error=f"{type(exc).__name__}: {exc}", started_at=start,
                                                     completed_at=datetime.now(UTC).isoformat(),
                                                     execution_order=order, attempt=attempt,
                                                     end_to_end_ms=(time.perf_counter() - tick) * 1000)
                        experiment.results.append(result)
                        if progress:
                            progress(experiment, result)
                allocation = parse_kv_allocation(server.read_logs())
                experiment.settings.setdefault("kv_allocation_by_config_context", {})[f"{ctx}:{config.id}"] = allocation
        except Exception as exc:  # noqa: BLE001 - report per-configuration runtime failures
            experiment.unsupported.append({"config_id": config.id, "context": ctx,
                                           "status": "runtime_failed" if stage == "runtime_probe" else "initialization_failed",
                                           "reason": f"{type(exc).__name__}: {exc}"})
            capability_records[f"{ctx}:{config.id}"] = {
                "status": "runtime_failed" if stage == "runtime_probe" else "initialization_failed",
                "reason": f"{type(exc).__name__}: {exc}"}
            if progress:
                progress(experiment, None)
    experiment.settings["capability_by_config_context"] = capability_records
    experiment.settings["execution_schedule"] = execution_schedule
    # Pair only results from identical context, fixture, and repetition; failed runs remain unavailable.
    experiment.pairings = pair_results(experiment.results, experiment.fixtures, baseline_config_id,
                                       [config.id for config in experiment.configurations], repetitions)
    if experiment.results and experiment.unsupported:
        experiment.status = "partial"
    elif not experiment.results:
        experiment.status = "failed"
    from hashlib import sha256
    reproducibility_settings = {key: experiment.settings.get(key) for key in (
        "contexts", "preset", "temperature", "seed", "gpu_layers", "flash_attention",
        "repetitions", "generation_max_tokens", "fixture_seed",
        "process_memory_sampling", "execution_schedule")}
    manifest = {"model_sha256": experiment.model_sha256,
                "runtime_sha256": experiment.runtime_sha256, "settings": reproducibility_settings,
                "fixtures": [f.model_dump(mode="json") for f in experiment.fixtures],
                "configurations": [c.model_dump(mode="json") for c in experiment.configurations]}
    import json
    experiment.manifest_sha256 = sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()
    if progress:
        progress(experiment, None)
    return experiment
