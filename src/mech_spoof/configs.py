"""Central configuration: model registry, paths, experiment constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    """Resolve repo root from env var or fall back to package location."""
    override = os.environ.get("MECH_SPOOF_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _repo_root()
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = Path(os.environ.get("MECH_SPOOF_RESULTS", REPO_ROOT / "results"))
CACHE_DIR = Path(os.environ.get("MECH_SPOOF_CACHE", REPO_ROOT / "cache"))


@dataclass(frozen=True)
class ModelConfig:
    """Static config for one model."""

    key: str
    hf_id: str
    template: str
    slug: str
    dtype: str = "bfloat16"
    best_layer_hint: int | None = None
    notes: str = ""

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.slug


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "qwen": ModelConfig(
        key="qwen",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        template="chatml",
        slug="qwen25_7b",
        dtype="bfloat16",
    ),
    "llama3": ModelConfig(
        key="llama3",
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        template="llama3",
        slug="llama3_8b",
        dtype="bfloat16",
    ),
    "mistral": ModelConfig(
        key="mistral",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        template="mistral",
        slug="mistral_7b",
        dtype="bfloat16",
    ),
    "gemma": ModelConfig(
        key="gemma",
        hf_id="google/gemma-2-9b-it",
        template="gemma",
        slug="gemma2_9b",
        dtype="bfloat16",
    ),
    "phi3": ModelConfig(
        key="phi3",
        hf_id="microsoft/Phi-3.5-mini-instruct",
        template="phi3",
        slug="phi35_mini",
        dtype="bfloat16",
    ),
    # Local smoke-test only (fits on 12 GB ROCm)
    "gemma_small": ModelConfig(
        key="gemma_small",
        hf_id="google/gemma-2-2b-it",
        template="gemma",
        slug="gemma2_2b",
        dtype="float16",
        notes="Local smoke-test model — too small for headline results.",
    ),
}


# ---------- Experiment constants ----------

STRUCTURAL_N_PER_CATEGORY = 100
STRUCTURAL_CATEGORIES = ["format", "persona", "behavioral", "style"]
STRUCTURAL_TRAIN_FRACTION = 0.75

CONFLICT_PAIRS_COUNT = 200
CONFLICT_CATEGORIES = {
    "language": 40,
    "length": 30,
    "format": 30,
    "topic": 30,
    "name": 20,
    "tone": 20,
    "string": 30,
}

PROBE_POSITION_STRATEGY = "response_first"  # see templates.base.TemplateAdapter
PROBE_LOGREG_C = 1.0
PROBE_TEST_SIZE = 0.25

HARMFUL_N = 100  # from AdvBench
HARMLESS_N = 100

GENERATION_MAX_NEW_TOKENS = 200
ATTACK_GENERATION_MAX_NEW_TOKENS = 50

BOOTSTRAP_N_RESAMPLES = 10_000
SEED = 42


@dataclass
class RunManifest:
    """Metadata captured alongside every result bundle."""

    model_key: str
    experiment: str
    timestamp: str
    git_sha: str | None = None
    gpu_name: str | None = None
    torch_version: str | None = None
    extras: dict = field(default_factory=dict)
