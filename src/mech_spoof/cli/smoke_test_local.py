"""Local smoke test: run an experiment on the default qwen model (Qwen3.5-4B).

Validates the whole pipeline — model loading, template adapter, activation extraction, probe
training — before spending Colab credits on the larger runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local smoke test (Qwen3.5-4B).")
    parser.add_argument("--model", default="qwen",
                        help="MODEL_CONFIGS key (default: qwen = Qwen/Qwen3.5-4B)")
    parser.add_argument("--experiment", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("results_smoke"))
    parser.add_argument("--n-instructions", type=int, default=10,
                        help="Sub-sample instructions per category for speed")
    args = parser.parse_args(argv)

    out_dir = args.out / args.model / f"exp{args.experiment}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == 1:
        # Monkey-patch the structural dataset size for speed
        from mech_spoof import configs
        configs.STRUCTURAL_N_PER_CATEGORY = args.n_instructions
        from mech_spoof.experiments.exp1_authority import run_experiment_1
        result = run_experiment_1(args.model, out_dir)
        print(f"[smoke] best layer: {result.best_layer}  best acc: {result.best_accuracy:.3f}")
        return 0

    if args.experiment == 2:
        from mech_spoof.experiments.exp2_conflict import run_experiment_2
        exp1_dir = args.out / args.model / "exp1"
        result = run_experiment_2(args.model, out_dir,
                                  exp1_dir=exp1_dir if exp1_dir.exists() else None)
        print(f"[smoke] conflict summary: {result.summary}")
        return 0

    if args.experiment == 3:
        exp1_dir = args.out / args.model / "exp1"
        if not exp1_dir.exists():
            print("Run experiment 1 first.")
            return 1
        from mech_spoof.experiments.exp3_refusal import run_experiment_3
        result = run_experiment_3(args.model, out_dir, exp1_dir=exp1_dir, n_harmful=20, n_harmless=20)
        print(f"[smoke] authority-refusal cos at best layer: {result.cosine_at_best_layer:.3f}")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
