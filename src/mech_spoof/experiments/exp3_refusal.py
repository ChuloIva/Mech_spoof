"""Experiment 3 — Refusal direction + authority-refusal geometry (§6)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mech_spoof.configs import HARMFUL_N, HARMLESS_N
from mech_spoof.datasets.advbench import load_advbench
from mech_spoof.datasets.harmless import load_harmless
from mech_spoof.directions import (
    GeometryReport,
    analyze_authority_refusal_relationship,
    compute_refusal_direction,
)
from mech_spoof.io import load_json, load_npz, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp3Result:
    model_key: str
    geometry: GeometryReport
    best_layer_authority: int | None
    cosine_at_best_layer: float | None


def _load_exp1_authority_dirs(exp1_dir: Path) -> tuple[int, dict[int, np.ndarray]]:
    result = load_json(Path(exp1_dir) / "result.json")
    best = int(result["best_layer"])
    arrays = load_npz(Path(exp1_dir) / "arrays.npz")
    auth_dirs = {}
    for k, v in arrays.items():
        if k.startswith("probe_dir_layer_"):
            layer = int(k.split("_")[-1])
            auth_dirs[layer] = v
    return best, auth_dirs


def run_experiment_3(
    model_key: str,
    out_dir: Path,
    exp1_dir: Path,
    n_harmful: int = HARMFUL_N,
    n_harmless: int = HARMLESS_N,
    seed: int = 42,
    free_after: bool = True,
) -> Exp3Result:
    """Compute refusal direction and its geometric relationship to the authority direction."""
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(exp1_dir).exists():
        raise FileNotFoundError(
            f"Exp3 requires Exp1 bundle at {exp1_dir} — run Experiment 1 first."
        )

    loaded = load_model(model_key)
    harmful = load_advbench()[:n_harmful]
    harmless = load_harmless(n=n_harmless)

    with timer(f"[{model_key}] compute refusal direction"):
        refusal_dirs = compute_refusal_direction(
            loaded,
            harmful,
            harmless,
            cache_dir=out_dir / "act_cache_refusal",
        )

    best_authority, auth_dirs = _load_exp1_authority_dirs(exp1_dir)
    geometry = analyze_authority_refusal_relationship(auth_dirs, refusal_dirs)

    arrays = {f"refusal_dir_layer_{l:03d}": v for l, v in refusal_dirs.items()}
    arrays["cosine_by_layer"] = np.array(
        [geometry.cosine_by_layer.get(l, float("nan")) for l in range(loaded.n_layers)]
    )
    arrays["principal_angle_deg_by_layer"] = np.array(
        [geometry.principal_angle_deg_by_layer.get(l, float("nan")) for l in range(loaded.n_layers)]
    )

    cos_at_best = geometry.cosine_by_layer.get(best_authority)

    result_json = {
        "model_key": model_key,
        "hf_id": loaded.cfg.hf_id,
        "n_layers": loaded.n_layers,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "best_layer_authority": int(best_authority),
        "cosine_at_best_authority_layer": float(cos_at_best) if cos_at_best is not None else None,
        "geometry": {
            "cosine_by_layer": {int(k): float(v) for k, v in geometry.cosine_by_layer.items()},
            "principal_angle_deg_by_layer": {
                int(k): float(v) for k, v in geometry.principal_angle_deg_by_layer.items()
            },
            "shared_variance_by_layer": {
                int(k): float(v) for k, v in geometry.shared_variance_by_layer.items()
            },
        },
    }
    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        manifest_extras={"experiment": "exp3_refusal", "model_key": model_key},
    )

    if free_after:
        free_model(loaded)

    return Exp3Result(
        model_key=model_key,
        geometry=geometry,
        best_layer_authority=best_authority,
        cosine_at_best_layer=cos_at_best,
    )
