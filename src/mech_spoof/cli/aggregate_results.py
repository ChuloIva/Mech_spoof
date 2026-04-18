"""Aggregate per-model result bundles into the cross-model report + paper figures."""

from __future__ import annotations

import argparse
from pathlib import Path

from mech_spoof.configs import RESULTS_DIR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=Path("results/_aggregate"))
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to aggregate (default: all)")
    args = parser.parse_args(argv)

    from mech_spoof.experiments.exp5_comparative import aggregate_results
    from mech_spoof.viz import (
        make_summary_table,
        plot_attack_prediction_scatter,
        plot_authority_refusal_cosine,
        plot_conflict_compliance_bars,
        plot_probe_accuracy_overlay,
        plot_token_trace,
    )

    args.out.mkdir(parents=True, exist_ok=True)

    report = aggregate_results(
        model_keys=args.models, results_root=args.results_root, out_dir=args.out,
    )
    per_model = report.per_model

    plot_probe_accuracy_overlay(per_model, args.out / "fig1_probe_accuracy_overlay.png")
    plot_authority_refusal_cosine(per_model, args.out / "fig2_authority_refusal_cosine.png")
    plot_conflict_compliance_bars(per_model, args.out / "fig3_conflict_compliance_bars.png")
    plot_attack_prediction_scatter(per_model, args.out / "fig4_attack_prediction_scatter.png")
    make_summary_table(per_model, args.out / "table1_summary.csv")

    # Token traces for any model that has them
    for model_key, data in per_model.items():
        exp4 = data.get("exp4_attacks") if isinstance(data, dict) else None
        if not exp4:
            continue
        for i, t in enumerate(exp4.get("traces", []) or []):
            plot_token_trace(
                t["trace"],
                args.out / f"fig5_trace_{model_key}_{i}.png",
                title=f"{model_key} — {t.get('harmful_goal', '')[:60]}",
            )

    print(f"[aggregate] wrote report to {args.out}")
    print(report.summary_table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
