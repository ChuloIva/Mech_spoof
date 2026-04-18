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
        hf_id="Qwen/Qwen3.5-4B",
        template="chatml",
        slug="qwen35_4b",
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
    # Smoke-test model. Not fitting in 12 GB bf16 locally — plan is Colab T4/A100 or
    # 4-bit local (bitsandbytes) for quick pipeline validation. Quantization distorts
    # activations so only use this for smoke-testing, not for headline results.
    "gemma_small": ModelConfig(
        key="gemma_small",
        hf_id="google/gemma-4-E4B-it",
        template="gemma4",
        slug="gemma4_e4b",
        dtype="bfloat16",
        notes=(
            "Gemma 4 E4B-it — 4.5B effective / ~8B total params. bf16 needs ~16 GB; "
            "use 4-bit (bitsandbytes) for 12 GB local or run on Colab T4+. "
            "New chat-template format uses <|turn> / <turn|> delimiters (not Gemma 2's "
            "<start_of_turn> / <end_of_turn>)."
        ),
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
