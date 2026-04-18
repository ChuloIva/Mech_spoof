"""Experiment 3 — Refusal direction + authority-refusal geometry (§6).

Default pipeline uses the OBLITERATUS prompt corpus (512 curated pairs) and the raw-text,
last-token, diff-of-means method from Arditi et al. 2024 — directly comparable with
OBLITERATUS results. Chat-template wrapping (`wrap_mode="chat"`) is available for those who
want to probe how refusal is encoded under the model's real inference-time prompt format.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mech_spoof.directions import (
    GeometryReport,
    WrapMode,
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
    strong_layers: list[int]
    source: str
    wrap_mode: WrapMode
    n_harmful: int
    n_harmless: int


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
    source: str = "builtin",
    wrap_mode: WrapMode = "raw",
    seed: int = 42,
    free_after: bool = True,
    n_harmful: int | None = None,
    n_harmless: int | None = None,
) -> Exp3Result:
    """Compute refusal direction via OBLITERATUS-style method and measure geometry vs authority.

    Parameters
    ----------
    source : str
        OBLITERATUS prompt source. One of
        "builtin" | "advbench" | "harmbench" | "anthropic_redteam" | "wildjailbreak".
    wrap_mode : "raw" | "chat"
        "raw" matches OBLITERATUS. "chat" wraps each prompt via the model's chat template.
    n_harmful, n_harmless : int, optional
        Cap the source prompt lists (useful on smaller GPUs).
    """
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(exp1_dir).exists():
        raise FileNotFoundError(
            f"Exp3 requires Exp1 bundle at {exp1_dir} — run Experiment 1 first."
        )

    loaded = load_model(model_key)

    # Optionally sub-sample
    harmful_list = None
    harmless_list = None
    if n_harmful or n_harmless:
        from mech_spoof.obliteratus_compat import load_prompt_pairs
        h, l = load_prompt_pairs(source)
        if n_harmful:
            h = h[: n_harmful]
        if n_harmless:
            l = l[: n_harmless]
        harmful_list, harmless_list = h, l

    with timer(f"[{model_key}] compute refusal direction (source={source}, wrap={wrap_mode})"):
        refusal = compute_refusal_direction(
            loaded,
            harmful=harmful_list,
            harmless=harmless_list,
            wrap_mode=wrap_mode,
            source=source,
            cache_dir=out_dir / "act_cache_refusal",
            select_strong_layers=True,
        )

    best_authority, auth_dirs = _load_exp1_authority_dirs(exp1_dir)
    geometry = analyze_authority_refusal_relationship(auth_dirs, refusal.directions)

    arrays = {f"refusal_dir_layer_{l:03d}": v for l, v in refusal.directions.items()}
    arrays["refusal_norms"] = np.array(
        [refusal.norms.get(l, float("nan")) for l in range(loaded.n_layers)]
    )
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
        "source": source,
        "wrap_mode": wrap_mode,
        "n_harmful": refusal.n_harmful,
        "n_harmless": refusal.n_harmless,
        "best_layer_authority": int(best_authority),
        "cosine_at_best_authority_layer": float(cos_at_best) if cos_at_best is not None else None,
        "strong_layers": refusal.strong_layers,
        "geometry": {
            "cosine_by_layer": {int(k): float(v) for k, v in geometry.cosine_by_layer.items()},
            "principal_angle_deg_by_layer": {
                int(k): float(v) for k, v in geometry.principal_angle_deg_by_layer.items()
            },
            "shared_variance_by_layer": {
                int(k): float(v) for k, v in geometry.shared_variance_by_layer.items()
            },
        },
        "refusal_norms": {int(k): float(v) for k, v in refusal.norms.items()},
    }
    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        manifest_extras={
            "experiment": "exp3_refusal", "model_key": model_key,
            "source": source, "wrap_mode": wrap_mode,
        },
    )

    if free_after:
        free_model(loaded)

    return Exp3Result(
        model_key=model_key,
        geometry=geometry,
        best_layer_authority=best_authority,
        cosine_at_best_layer=cos_at_best,
        strong_layers=refusal.strong_layers,
        source=source,
        wrap_mode=wrap_mode,
        n_harmful=refusal.n_harmful,
        n_harmless=refusal.n_harmless,
    )
