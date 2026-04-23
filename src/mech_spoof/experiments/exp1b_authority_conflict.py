"""Experiment 1b — Authority direction from conflict pairs.

Uses evolved conflict pairs with gold responses to find the authority direction
from *behavioral* differences rather than structural position alone. Each pair
produces 4 traces (2 orderings × 2 response alignments) with role-swapping to
deconfound instruction content from the authority label.

Pipeline:
    1. Load model.
    2. Load stratified conflict pairs, build 4-trace contrastive bundles.
    3. Extract activations at last token (end of gold response) for all traces.
    4. Train per-layer logistic probes + difference-in-means directions.
    5. Optionally compare with Exp 1 structural direction.
    6. Save probes, directions, accuracies, and per-axis breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mech_spoof.activations import extract_at_last_token_batched
from mech_spoof.configs import MODEL_CONFIGS
from mech_spoof.datasets.conflict_evolved import (
    build_conflict_traces,
    flatten_traces,
    load_evolved_conflict_pairs,
)
from mech_spoof.io import load_result_bundle, save_pickle, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.probes import (
    compute_authority_direction_dim,
    cosine_agreement,
    find_best_layer,
    train_authority_probe,
)
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp1bResult:
    model_key: str
    n_layers: int
    d_model: int
    n_pairs: int
    n_system_following: int
    n_user_following: int
    accuracies: dict[int, float]
    aurocs: dict[int, float]
    best_layer: int
    best_accuracy: float
    probe_vs_dim_cosine: dict[int, float]
    exp1_cosine_agreement: dict[int, float] | None
    macro_axis_breakdown: dict[str, dict]


def _load_exp1_directions(exp1_dir: Path) -> dict[int, np.ndarray] | None:
    """Load exp1 probe directions if available. Returns None if not found."""
    try:
        bundle = load_result_bundle(exp1_dir)
    except FileNotFoundError:
        return None
    arrays = bundle.get("arrays", {})
    dirs = {}
    for key in arrays:
        if key.startswith("probe_dir_layer_"):
            layer = int(key.split("_")[-1])
            d = arrays[key]
            norm = np.linalg.norm(d)
            if norm > 1e-8:
                dirs[layer] = d / norm
    return dirs if dirs else None


def run_experiment_1b(
    model_key: str,
    out_dir: Path,
    exp1_dir: Path | None = None,
    seed: int = 42,
    free_after: bool = True,
    cache_activations: bool = True,
    batch_size: int = 4,
    max_length: int = 2048,
    max_response_chars: int = 3000,
    max_pairs: int | None = None,
) -> Exp1bResult:
    """Run Experiment 1b end-to-end for one model."""
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(model_key)
    logger.info(
        f"[{model_key}] n_layers={loaded.n_layers} d_model={loaded.d_model}"
    )

    # Build dataset
    with timer(f"[{model_key}] load + build conflict traces"):
        pairs = load_evolved_conflict_pairs(max_pairs=max_pairs)
        quads = build_conflict_traces(
            loaded.tokenizer,
            pairs,
            supports_enable_thinking=getattr(
                loaded.template, "_supports_enable_thinking", False
            ),
            system_role_supported=getattr(
                loaded.template, "_system_role_supported", True
            ),
            max_response_chars=max_response_chars,
        )
        sys_bundles, usr_bundles = flatten_traces(quads)

    logger.info(
        f"[{model_key}] {len(sys_bundles)} system-following / "
        f"{len(usr_bundles)} user-following traces"
    )

    # Extract activations
    cache_s = out_dir / "act_cache_sys_follow" if cache_activations else None
    cache_u = out_dir / "act_cache_usr_follow" if cache_activations else None

    with timer(f"[{model_key}] extract sys-following activations (bs={batch_size})"):
        sys_acts = extract_at_last_token_batched(
            loaded,
            sys_bundles,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_s,
        )
    with timer(f"[{model_key}] extract usr-following activations (bs={batch_size})"):
        usr_acts = extract_at_last_token_batched(
            loaded,
            usr_bundles,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_u,
        )

    # Train probes + diff-in-means
    with timer(f"[{model_key}] train probes"):
        probe_res = train_authority_probe(sys_acts, usr_acts, seed=seed)
    dim_dirs = compute_authority_direction_dim(sys_acts, usr_acts)
    agree = cosine_agreement(probe_res.directions, dim_dirs)
    best = find_best_layer(probe_res.accuracies)

    logger.info(
        f"[{model_key}] best_layer={best} "
        f"accuracy={probe_res.accuracies[best]:.4f} "
        f"auroc={probe_res.aurocs[best]:.4f}"
    )

    # Cross-check against exp1 if available
    exp1_cosine = None
    if exp1_dir is not None:
        exp1_dirs = _load_exp1_directions(Path(exp1_dir))
        if exp1_dirs is not None:
            exp1_cosine = cosine_agreement(probe_res.directions, exp1_dirs)
            cos_at_best = exp1_cosine.get(best, float("nan"))
            logger.info(
                f"[{model_key}] exp1 cosine at best layer: {cos_at_best:.4f}"
            )

    # Per-macro-axis breakdown at best layer
    from sklearn.metrics import accuracy_score, roc_auc_score

    all_bundles = sys_bundles + usr_bundles
    all_labels = np.array([1] * len(sys_bundles) + [0] * len(usr_bundles))
    all_acts = np.concatenate([sys_acts, usr_acts], axis=0)

    test_idx = probe_res.test_indices
    X_test_best = all_acts[test_idx, best, :]
    norms = np.linalg.norm(X_test_best, axis=1, keepdims=True) + 1e-8
    X_test_best = X_test_best / norms
    y_test = all_labels[test_idx]

    probe = probe_res.probes[best]
    y_pred = probe.predict(X_test_best)
    y_prob = probe.predict_proba(X_test_best)[:, 1]

    macro_axis_breakdown = {}
    for i_test, global_idx in enumerate(test_idx):
        axis = all_bundles[global_idx].extras["macro_axis"]
        if axis not in macro_axis_breakdown:
            macro_axis_breakdown[axis] = {"y_true": [], "y_pred": [], "y_prob": []}
        macro_axis_breakdown[axis]["y_true"].append(y_test[i_test])
        macro_axis_breakdown[axis]["y_pred"].append(y_pred[i_test])
        macro_axis_breakdown[axis]["y_prob"].append(y_prob[i_test])

    axis_results = {}
    for axis, data in sorted(macro_axis_breakdown.items()):
        n = len(data["y_true"])
        yt = np.array(data["y_true"])
        yp = np.array(data["y_pred"])
        yprob = np.array(data["y_prob"])
        acc = float(accuracy_score(yt, yp))
        try:
            auc = float(roc_auc_score(yt, yprob))
        except ValueError:
            auc = float("nan")
        axis_results[axis] = {"n": n, "accuracy": acc, "auroc": auc}
        logger.info(f"  {axis:15s}  n={n:4d}  acc={acc:.3f}  auc={auc:.3f}")

    # Save
    arrays = {
        f"probe_dir_layer_{l:03d}": v for l, v in probe_res.directions.items()
    }
    arrays.update({f"dim_dir_layer_{l:03d}": v for l, v in dim_dirs.items()})
    arrays["accuracies"] = np.array(
        [probe_res.accuracies[i] for i in range(loaded.n_layers)]
    )
    arrays["aurocs"] = np.array(
        [probe_res.aurocs[i] for i in range(loaded.n_layers)]
    )

    result_json = {
        "model_key": model_key,
        "hf_id": loaded.cfg.hf_id,
        "n_layers": loaded.n_layers,
        "d_model": loaded.d_model,
        "n_pairs": len(pairs),
        "n_system_following": len(sys_bundles),
        "n_user_following": len(usr_bundles),
        "accuracies": {int(k): float(v) for k, v in probe_res.accuracies.items()},
        "aurocs": {int(k): float(v) for k, v in probe_res.aurocs.items()},
        "best_layer": int(best),
        "best_accuracy": float(probe_res.accuracies[best]),
        "probe_vs_dim_cosine": {int(k): float(v) for k, v in agree.items()},
        "exp1_cosine_agreement": (
            {int(k): float(v) for k, v in exp1_cosine.items()}
            if exp1_cosine
            else None
        ),
        "macro_axis_breakdown": axis_results,
        "test_idx": probe_res.test_indices.tolist(),
    }

    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        pickles={"probes": probe_res.probes},
        manifest_extras={
            "experiment": "exp1b_authority_conflict",
            "model_key": model_key,
        },
    )
    save_pickle(probe_res, out_dir / "probe_result.pkl")

    if free_after:
        free_model(loaded)

    return Exp1bResult(
        model_key=model_key,
        n_layers=loaded.n_layers,
        d_model=loaded.d_model,
        n_pairs=len(pairs),
        n_system_following=len(sys_bundles),
        n_user_following=len(usr_bundles),
        accuracies=probe_res.accuracies,
        aurocs=probe_res.aurocs,
        best_layer=best,
        best_accuracy=probe_res.accuracies[best],
        probe_vs_dim_cosine=agree,
        exp1_cosine_agreement=exp1_cosine,
        macro_axis_breakdown=axis_results,
    )
