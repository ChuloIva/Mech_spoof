"""Experiment 5 — Cross-model comparative aggregation (§8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mech_spoof.configs import MODEL_CONFIGS, RESULTS_DIR
from mech_spoof.io import load_json, load_npz, save_json
from mech_spoof.utils import get_logger

logger = get_logger(__name__)


@dataclass
class CrossModelReport:
    per_model: dict[str, dict]
    summary_table: pd.DataFrame


def _safe_load(path: Path):
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception as e:
        logger.warning(f"failed to load {path}: {e}")
        return None


def _load_per_model(model_key: str, results_root: Path) -> dict:
    slug = MODEL_CONFIGS[model_key].slug
    base = results_root / slug
    out: dict = {"model_key": model_key, "slug": slug}
    for exp in ("exp1_authority", "exp2_conflict", "exp3_refusal", "exp4_attacks"):
        j = _safe_load(base / exp / "result.json")
        if j is not None:
            out[exp] = j
    # Arrays (per-layer curves)
    arr_dir = base / "exp1_authority" / "arrays.npz"
    if arr_dir.exists():
        out["exp1_arrays"] = load_npz(arr_dir)
    arr_dir3 = base / "exp3_refusal" / "arrays.npz"
    if arr_dir3.exists():
        out["exp3_arrays"] = load_npz(arr_dir3)
    return out


def aggregate_results(
    model_keys: list[str] | None = None,
    results_root: Path = RESULTS_DIR,
    out_dir: Path | None = None,
) -> CrossModelReport:
    """Load per-model bundles and build the paper summary table + intermediate per-model dict."""
    model_keys = model_keys or list(MODEL_CONFIGS.keys())
    per_model = {k: _load_per_model(k, results_root) for k in model_keys}

    rows = []
    for k, data in per_model.items():
        row = {"model": k, "slug": MODEL_CONFIGS[k].slug}
        exp1 = data.get("exp1_authority", {})
        row.update({
            "n_layers": exp1.get("n_layers"),
            "d_model": exp1.get("d_model"),
            "exp1_best_layer": exp1.get("best_layer"),
            "exp1_best_accuracy": exp1.get("best_accuracy"),
            "fake_delim_collapsed": (
                exp1.get("fake_delim_report", {}).get("any_collapsed")
                if exp1 else None
            ),
        })
        exp3 = data.get("exp3_refusal", {})
        row["exp3_cosine_at_best_layer"] = exp3.get("cosine_at_best_authority_layer")
        exp2 = data.get("exp2_conflict", {})
        if exp2:
            summary = exp2.get("summary", {})
            for cond in ("REAL", "NONE", "FAKE"):
                row[f"exp2_{cond}_compliance"] = summary.get(cond, {}).get("compliance_rate")
            corr = exp2.get("correlation", {})
            if "overall" in corr:
                row["exp2_probe_behavior_r"] = corr["overall"]["r"]
        exp4 = data.get("exp4_attacks", {})
        if exp4:
            attack_summary = exp4.get("summary", {})
            for atype, stats in attack_summary.items():
                row[f"exp4_{atype}_refusal_rate"] = stats.get("refusal_rate")
        rows.append(row)

    df = pd.DataFrame(rows)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "summary_table.csv", index=False)
        save_json({"per_model_keys": model_keys, "rows": rows}, out_dir / "summary.json")

    return CrossModelReport(per_model=per_model, summary_table=df)
