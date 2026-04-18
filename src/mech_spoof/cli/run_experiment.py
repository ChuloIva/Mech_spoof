"""Run one experiment for one model, saving the result bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from mech_spoof.configs import MODEL_CONFIGS, RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    cfg = MODEL_CONFIGS[args.model]
    base = args.results_root / cfg.slug
    exp_name = {
        1: "exp1_authority", 2: "exp2_conflict", 3: "exp3_refusal", 4: "exp4_attacks",
    }[args.experiment]
    out_dir = base / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == 1:
        from mech_spoof.experiments.exp1_authority import run_experiment_1
        run_experiment_1(args.model, out_dir)
    elif args.experiment == 2:
        from mech_spoof.experiments.exp2_conflict import run_experiment_2
        run_experiment_2(args.model, out_dir, exp1_dir=base / "exp1_authority")
    elif args.experiment == 3:
        from mech_spoof.experiments.exp3_refusal import run_experiment_3
        run_experiment_3(args.model, out_dir, exp1_dir=base / "exp1_authority")
    elif args.experiment == 4:
        from mech_spoof.experiments.exp4_attacks import run_experiment_4
        run_experiment_4(
            args.model, out_dir,
            exp1_dir=base / "exp1_authority",
            exp3_dir=base / "exp3_refusal",
        )
    print(f"[done] {args.model} exp{args.experiment} -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
