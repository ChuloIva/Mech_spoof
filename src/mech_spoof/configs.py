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
    # Optional bitsandbytes quantization. None = full precision (per `dtype`).
    # "8bit" → BitsAndBytesConfig(load_in_8bit=True); "4bit" → load_in_4bit=True (NF4).
    # When set, load_model uses device_map="auto" (skips manual .to(device)) so
    # the model can span multiple GPUs / offload — required for very large models.
    quantization: str | None = None

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
    "llama33_70b": ModelConfig(
        key="llama33_70b",
        hf_id="meta-llama/Llama-3.3-70B-Instruct",
        template="llama3",
        slug="llama33_70b",
        dtype="bfloat16",
        quantization="8bit",
        notes=(
            "Llama 3.3 70B Instruct, loaded 8-bit via bitsandbytes (~70 GB VRAM, "
            "fits A100 80GB / H100 80GB single-GPU). Reuses the llama3 template "
            "(same <|start_header_id|>...<|end_header_id|> chat format). "
            "Target for the cross-direction comparison demo: pre-computed "
            "Goodfire SAE at L50 (Goodfire/Llama-3.3-70B-Instruct-SAE-l50, on "
            "Neuronpedia w/ auto-interp), pre-computed Assistant Axis at "
            "lu-christina/assistant-axis-vectors/llama-3.3-70b/, and refusal "
            "direction from huihui-ai/Llama-3.3-70B-Instruct-abliterated. "
            "8-bit quantization adds noise vs the bf16 activations Goodfire's "
            "SAE was trained on — direction-fitting still works, but SAE "
            "feature-projection accuracy degrades modestly."
        ),
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
    "gemma3_4b": ModelConfig(
        key="gemma3_4b",
        hf_id="google/gemma-3-4b-it",
        template="gemma",
        slug="gemma3_4b",
        dtype="bfloat16",
        notes=(
            "Gemma 3 4B Instruction-Tuned. Composite multimodal config "
            "(Gemma3ForConditionalGeneration); model loader's composite path handles it. "
            "Same <start_of_turn>/<end_of_turn> template as Gemma 2 — reuses the gemma adapter. "
            "CAVEAT for exp06: Gemma's template folds system content into the first user turn, "
            "so S and U conditions differ only by a newline at the token level. Direction "
            "fitted on this model captures structural punctuation, not role authority — useful "
            "as a negative control, not as a headline result. "
            "Has Gemma Scope 2 SAEs + transcoders + crosscoders on Neuronpedia."
        ),
    ),
    "qwen3_4b": ModelConfig(
        key="qwen3_4b",
        hf_id="Qwen/Qwen3-4B",
        template="chatml",
        slug="qwen3_4b",
        dtype="bfloat16",
        notes=(
            "Qwen3-4B (NOT Qwen3.5-4B). Plain text decoder, 36 layers, d_model=2560, "
            "ChatML template with native system role. Refit target for SAE-projection "
            "demo: Hanna & Piotrowski transcoders at https://hf.co/mwhanna/qwen3-4b-transcoders "
            "(layer_0..layer_35.safetensors, ~1.68 GB each, 164k MLP-output features). "
            "Auto-interp on Neuronpedia at /qwen3-4b/{0,23,30}-transcoder-hp."
        ),
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
