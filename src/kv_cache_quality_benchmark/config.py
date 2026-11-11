"""Pydantic-validated YAML experiment configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import KVConfig


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(StrictModel):
    binary: str = "llama-server"
    model: Path | None = None
    gpu_layers: int | Literal["all"] = "all"
    flash_attention: str = "on"
    parallel: int = 1
    prompt_cache: bool = False
    context_shift: bool = False
    auto_fit: bool = False


class GenerationConfig(StrictModel):
    temperature: float = 0
    max_output_tokens: int = Field(default=64, ge=1)
    seed: int = 42


class MeasurementConfig(StrictModel):
    process_memory: bool = True
    sample_interval_ms: int = Field(default=100, ge=10)


class ExperimentFile(StrictModel):
    name: str = "local"
    seed: int = 42
    repetitions: int = Field(default=1, ge=1, le=100)
    preset: Literal["smoke", "standard", "extended"] = "smoke"


class AppConfig(StrictModel):
    experiment: ExperimentFile = Field(default_factory=ExperimentFile)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    contexts: list[int] = Field(default_factory=list)
    cache_configs: list[KVConfig] = Field(default_factory=list)
    workloads: list[str] = Field(default_factory=list)
    measurement: MeasurementConfig = Field(default_factory=MeasurementConfig)

    @model_validator(mode="after")
    def unique_cache_config_ids(self) -> AppConfig:
        ids = [config.id for config in self.cache_configs]
        if len(ids) != len(set(ids)):
            raise ValueError("cache_configs contains duplicate IDs")
        if any(context < 256 for context in self.contexts):
            raise ValueError("context lengths must be at least 256 tokens")
        if self.runtime.parallel != 1:
            raise ValueError("controlled primary experiments require runtime.parallel = 1")
        if self.runtime.prompt_cache or self.runtime.context_shift or self.runtime.auto_fit:
            raise ValueError("controlled experiments require prompt_cache, context_shift, and auto_fit to be false")
        return self


def load_config(path: Path) -> AppConfig:
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to read YAML config {path}: {exc}") from exc
    config = AppConfig.model_validate(raw)
    base = path.resolve().parent
    if config.runtime.model and not config.runtime.model.is_absolute():
        config.runtime.model = (base / config.runtime.model).resolve()
    binary_path = Path(config.runtime.binary)
    if ("/" in config.runtime.binary or binary_path.suffix) and not binary_path.is_absolute():
        config.runtime.binary = str((base / binary_path).resolve())
    return config
