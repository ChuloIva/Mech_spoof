"""Experiment 2 — Conflict behavioral test (§5).

For each conflict pair, generate responses under REAL / NONE / FAKE conditions, evaluate whether
the model followed the system instruction, and record the probe score at response_first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mech_spoof.activations import response_first_position
from mech_spoof.configs import GENERATION_MAX_NEW_TOKENS
from mech_spoof.datasets.conflicting import build_conflicting_pairs
from mech_spoof.eval.compliance import evaluate_compliance
from mech_spoof.io import load_npz, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp2Result:
    model_key: str
    n_pairs: int
    summary: dict          # per-condition compliance rate + mean probe score
    correlation: dict      # per-condition and overall Pearson r


def _load_exp1_best_direction(exp1_dir: Path) -> tuple[int, np.ndarray] | None:
    """Load best-layer authority direction from exp1 bundle (probe direction)."""
    if exp1_dir is None or not Path(exp1_dir).exists():
        return None
    from mech_spoof.io import load_json
    result = load_json(Path(exp1_dir) / "result.json")
    best = int(result["best_layer"])
    arrays = load_npz(Path(exp1_dir) / "arrays.npz")
    key = f"probe_dir_layer_{best:03d}"
    return best, arrays[key]


def _score_residual_at_position(loaded, input_ids, layer: int, position: int,
                                direction: np.ndarray) -> float:
    """Extract the residual stream at (layer, position) and project onto `direction`."""
    import torch

    storage: list = [None] * loaded.n_layers
    handles = []
    for i in range(loaded.n_layers):
        def _hook(_m, _inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            storage[idx] = h.detach()
        handles.append(loaded.layer_module(i).register_forward_hook(_hook))

    try:
        ids = torch.tensor([input_ids], dtype=torch.long, device=loaded.device)
        with torch.no_grad():
            loaded.hf_model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    act = storage[layer][0, position].float().cpu().numpy()
    act = act / (np.linalg.norm(act) + 1e-8)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return float(act @ direction)


def run_experiment_2(
    model_key: str,
    out_dir: Path,
    exp1_dir: Optional[Path] = None,
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
    seed: int = 42,
    free_after: bool = True,
) -> Exp2Result:
    """Run Experiment 2 end-to-end for one model."""
    import torch
    from scipy.stats import pearsonr

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(model_key)

    # Optional authority probe for correlation analysis
    best_dir = _load_exp1_best_direction(exp1_dir) if exp1_dir else None
    if best_dir is None:
        logger.warning(
            f"[{model_key}] exp1 bundle not found — skipping probe-score correlation"
        )
    best_layer, direction = (best_dir or (None, None))

    conflict_prompts = build_conflicting_pairs(loaded.template)
    logger.info(f"[{model_key}] conflict pairs: {len(conflict_prompts)}")

    from tqdm.auto import tqdm

    rows: list[dict] = []
    for idx, cp in enumerate(tqdm(conflict_prompts, desc=f"exp2 {model_key}")):
        for condition_name, bundle in (("REAL", cp.real), ("NONE", cp.none), ("FAKE", cp.fake)):
            ids = torch.tensor([bundle.input_ids], dtype=torch.long, device=loaded.device)
            with torch.no_grad():
                output = loaded.hf_model.generate(
                    ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                )
            response = loaded.tokenizer.decode(
                output[0][ids.shape[1]:], skip_special_tokens=True
            )

            system_followed = evaluate_compliance(response, cp.pair, which="system")

            probe_score = None
            if direction is not None:
                probe_score = _score_residual_at_position(
                    loaded, bundle.input_ids, best_layer,
                    response_first_position(loaded, bundle), direction,
                )

            rows.append({
                "pair_id": cp.pair.id,
                "category": cp.pair.category,
                "eval_type": cp.pair.eval,
                "condition": condition_name,
                "system_followed": bool(system_followed),
                "probe_score": probe_score,
                "response": response,
            })

    # Aggregate
    summary: dict = {}
    for cond in ("REAL", "NONE", "FAKE"):
        subset = [r for r in rows if r["condition"] == cond]
        comply = np.mean([1 if r["system_followed"] else 0 for r in subset])
        scores = [r["probe_score"] for r in subset if r["probe_score"] is not None]
        summary[cond] = {
            "compliance_rate": float(comply),
            "mean_probe_score": float(np.mean(scores)) if scores else None,
            "std_probe_score": float(np.std(scores)) if scores else None,
            "n": len(subset),
        }

    correlation: dict = {}
    if direction is not None:
        all_scores = np.array([r["probe_score"] for r in rows if r["probe_score"] is not None])
        all_labels = np.array([
            1 if r["system_followed"] else 0
            for r in rows if r["probe_score"] is not None
        ])
        if len(all_scores) > 2 and len(np.unique(all_labels)) > 1:
            r_val, p_val = pearsonr(all_scores, all_labels)
            correlation["overall"] = {"r": float(r_val), "p": float(p_val), "n": int(len(all_scores))}
        for cond in ("REAL", "NONE", "FAKE"):
            subset = [r for r in rows
                      if r["condition"] == cond and r["probe_score"] is not None]
            if len(subset) > 2:
                xs = np.array([r["probe_score"] for r in subset])
                ys = np.array([1 if r["system_followed"] else 0 for r in subset])
                if len(np.unique(ys)) > 1:
                    r_val, p_val = pearsonr(xs, ys)
                    correlation[cond] = {"r": float(r_val), "p": float(p_val), "n": len(subset)}

    result_json = {
        "model_key": model_key,
        "n_pairs": len(conflict_prompts),
        "summary": summary,
        "correlation": correlation,
        "rows": rows,
    }
    save_result_bundle(
        out_dir,
        json_obj=result_json,
        manifest_extras={"experiment": "exp2_conflict", "model_key": model_key},
    )

    if free_after:
        free_model(loaded)

    return Exp2Result(
        model_key=model_key,
        n_pairs=len(conflict_prompts),
        summary=summary,
        correlation=correlation,
    )
