"""Process memory sampling and conservative llama.cpp KV log extraction."""
from __future__ import annotations

import re
import threading
from typing import Any, Self

import psutil


class ProcessMemorySampler:
    def __init__(self, pid: int, interval_ms: int = 100) -> None:
        self.process = psutil.Process(pid)
        self.interval = max(10, interval_ms) / 1000
        self.initial: int | None = None
        self.peak: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                value = self.process.memory_info().rss
                self.peak = max(self.peak or value, value)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> Self:
        try:
            self.initial = self.process.memory_info().rss
            self.peak = self.initial
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def parse_kv_allocation(log: str) -> dict[str, Any]:
    """Extract only explicit KV buffer sizes; never infer allocation from model dimensions."""
    pattern = re.compile(r"(?i)(?:KV|K\s*/\s*V|K\s+cache|V\s+cache).*?(?:size|buffer)\s*[=:]\s*([\d.]+)\s*(bytes?|KiB|MiB|GiB|KB|MB|GB)")
    multiplier = {"byte": 1, "bytes": 1, "kib": 1024, "kb": 1000, "mib": 1024**2,
                  "mb": 1000**2, "gib": 1024**3, "gb": 1000**3}
    found: list[dict[str, Any]] = []
    component_pattern = re.compile(r"(?i)([KV])\s*\([^)]*\)\s*:\s*([\d.]+)\s*(bytes?|KiB|MiB|GiB|KB|MB|GB)")
    for line in log.splitlines():
        components = component_pattern.findall(line)
        if components:
            for name, number, unit in components:
                found.append({"component": name.upper(), "line": line.strip(),
                              "bytes": int(float(number) * multiplier[unit.casefold()])})
            continue
        match = pattern.search(line)
        if match:
            number, unit = match.group(1), match.group(2)
            if number is not None and unit is not None:
                found.append({"line": line.strip(), "bytes": int(float(number) * multiplier[unit.casefold()])})
    return {"available": bool(found), "allocations": found,
            "total_bytes": sum(x["bytes"] for x in found) if found else None,
            "provenance": "parsed explicit llama.cpp log line" if found else "unavailable"}


def parse_context_metadata(log: str) -> dict[str, int | None]:
    """Read explicitly labeled context values; absent metadata stays unknown."""
    effective = re.search(r"(?im)\b(?:n_ctx_per_seq|n_ctx|ctx[-_ ]size)\s*[=:]\s*(\d+)", log)
    trained = re.search(r"(?im)\bn_ctx_train\s*[=:]\s*(\d+)", log)
    return {"effective_context": int(effective.group(1)) if effective else None,
            "training_context": int(trained.group(1)) if trained else None}


def response_timings(response: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    result: dict[str, Any] = {"end_to_end_ms": elapsed_ms}
    raw = response.get("timings")
    if isinstance(raw, dict):
        result["llama_cpp"] = raw
        for source, dest in (("prompt_per_second", "prompt_tokens_per_second"),
                             ("predicted_per_second", "generation_tokens_per_second"),
                             ("prompt_ms", "prompt_processing_ms"),
                             ("predicted_ms", "generation_ms"), ("predicted_n", "generated_tokens")):
            if source in raw:
                result[dest] = raw[source]
    return result
