"""Build or refresh the committed dataset JSONs.

Two backends:
- vllm   — default; runs a local HF instruct model via vLLM (Colab-friendly, no API bills)
- claude — uses the Anthropic API (needs ANTHROPIC_API_KEY)

Reads seed data, expands each category to the target count, writes
`data/structural/instructions_<cat>.json` and `data/conflicting_pairs.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mech_spoof.configs import (
    CONFLICT_CATEGORIES,
    CONFLICT_PAIRS_COUNT,
    DATA_DIR,
    STRUCTURAL_CATEGORIES,
    STRUCTURAL_N_PER_CATEGORY,
)
from mech_spoof.datasets.advbench import load_advbench
from mech_spoof.datasets.harmless import write_harmless_json
from mech_spoof.datasets.structural import _load_seeds
from mech_spoof.utils import get_logger

logger = get_logger(__name__)


_DEFAULT_MODELS = {
    "vllm": "Qwen/Qwen2.5-7B-Instruct",
    "claude": "claude-sonnet-4-6",
}


def _pick_generator(backend: str):
    if backend == "vllm":
        from mech_spoof.datasets import _vllm_gen as gen
    elif backend == "claude":
        from mech_spoof.datasets import _claude_gen as gen
    else:
        raise ValueError(f"Unknown backend {backend!r}. Use 'vllm' or 'claude'.")
    return gen


def expand_structural(
    target_per_category: int,
    backend: str,
    model: str,
    data_dir: Path = DATA_DIR,
    engine_kwargs: dict | None = None,
) -> None:
    gen = _pick_generator(backend)
    seeds = _load_seeds(data_dir)
    out_dir = data_dir / "structural"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in STRUCTURAL_CATEGORIES:
        have = seeds.get(cat, [])
        need = max(0, target_per_category - len(have))
        if need == 0:
            items = have[:target_per_category]
        else:
            kwargs = {"engine_kwargs": engine_kwargs} if backend == "vllm" else {}
            new = gen.generate_instructions(cat, have, n=need, model=model, **kwargs)
            items = list(dict.fromkeys(have + new))[:target_per_category]
        out_path = out_dir / f"instructions_{cat}.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        logger.info(f"[{cat}] wrote {len(items)} items -> {out_path}")


def expand_conflicts(
    target_count: int,
    backend: str,
    model: str,
    data_dir: Path = DATA_DIR,
    engine_kwargs: dict | None = None,
) -> None:
    gen = _pick_generator(backend)
    seed_path = data_dir / "conflicting_pairs_seed.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing {seed_path}")
    with open(seed_path) as f:
        seeds = json.load(f)

    by_cat: dict[str, list[dict]] = {}
    for p in seeds:
        by_cat.setdefault(p["category"], []).append(p)

    allowed_evals = [
        "language_check", "length_check", "format_check", "topic_check",
        "name_check", "tone_check", "string_check",
    ]
    eval_for_cat = {
        "language": "language_check",
        "length": "length_check",
        "format": "format_check",
        "topic": "topic_check",
        "name": "name_check",
        "tone": "tone_check",
        "string": "string_check",
    }

    out_pairs: list[dict] = []
    for cat, target in CONFLICT_CATEGORIES.items():
        have = by_cat.get(cat, [])
        need = max(0, target - len(have))
        if need == 0:
            out_pairs.extend(have[:target])
            continue
        kwargs = {"engine_kwargs": engine_kwargs} if backend == "vllm" else {}
        new = gen.generate_conflict_pairs(
            category=cat, eval_type=eval_for_cat[cat],
            allowed_evals=allowed_evals, examples=have, n=need, model=model,
            **kwargs,
        )
        for i, p in enumerate(new, start=len(have) + 1):
            p.setdefault("id", f"{cat[:4]}_{i:02d}")
            p.setdefault("category", cat)
            p.setdefault("eval", eval_for_cat[cat])
        combined = have + new
        out_pairs.extend(combined[:target])

    out_path = data_dir / "conflicting_pairs.json"
    with open(out_path, "w") as f:
        json.dump(out_pairs, f, indent=2)
    logger.info(f"wrote {len(out_pairs)} conflict pairs -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "claude"], default="vllm")
    parser.add_argument("--model", default=None,
                        help="HF model id (vllm) or Anthropic model name (claude). "
                             "Defaults per backend.")
    parser.add_argument("--structural-n", type=int, default=STRUCTURAL_N_PER_CATEGORY)
    parser.add_argument("--conflict-n", type=int, default=CONFLICT_PAIRS_COUNT)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-conflicts", action="store_true")
    parser.add_argument("--skip-advbench", action="store_true")
    parser.add_argument("--vllm-max-model-len", type=int, default=4096,
                        help="vLLM engine max_model_len")
    parser.add_argument("--vllm-gpu-mem", type=float, default=0.90,
                        help="vLLM engine gpu_memory_utilization")
    args = parser.parse_args(argv)

    model = args.model or _DEFAULT_MODELS[args.backend]
    engine_kwargs = None
    if args.backend == "vllm":
        engine_kwargs = {
            "max_model_len": args.vllm_max_model_len,
            "gpu_memory_utilization": args.vllm_gpu_mem,
        }

    if not args.skip_structural:
        expand_structural(args.structural_n, args.backend, model,
                          data_dir=args.data_dir, engine_kwargs=engine_kwargs)

    if not args.skip_conflicts:
        expand_conflicts(args.conflict_n, args.backend, model,
                         data_dir=args.data_dir, engine_kwargs=engine_kwargs)

    if not args.skip_advbench:
        goals = load_advbench()
        logger.info(f"AdvBench: {len(goals)} harmful goals cached.")

    harmless_path = write_harmless_json(args.data_dir)
    logger.info(f"harmless set -> {harmless_path}")

    if args.backend == "vllm":
        from mech_spoof.datasets._vllm_gen import free_llm
        free_llm()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
