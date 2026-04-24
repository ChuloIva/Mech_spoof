"""Experiment 1b — Authority direction from conflict pairs.

Uses evolved conflict pairs with gold responses to find the authority direction
from *behavioral* differences rather than structural position alone. Each pair
produces 4 traces (2 orderings × 2 response alignments) with role-swapping to
deconfound instruction content from the authority label.

Pipeline:
    1. Load model.
    2. Load stratified conflict pairs, build 4-trace contrastive bundles.
    3. Extract residual activations at 3 positions (response_first, mid, last) AND
       compute perplexity over the gold response span in one forward pass.
    4. Train per-layer logistic probes + diff-in-means directions at each position.
    5. Optionally compare with Exp 1 structural direction.
    6. Save probes, directions, per-position per-layer accuracies, per-prompt
       metadata table (for the Streamlit viewer), and per-axis breakdown.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mech_spoof.activations import extract_multi_position_with_ppl_batched
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

DEFAULT_POSITIONS: tuple[str, ...] = ("response_first", "response_mid", "response_last")


@dataclass
class Exp1bResult:
    model_key: str
    n_layers: int
    d_model: int
    n_pairs: int
    n_system_following: int
    n_user_following: int
    positions: tuple[str, ...]
    accuracies: dict[str, dict[int, float]]          # position -> layer -> acc
    aurocs: dict[str, dict[int, float]]              # position -> layer -> auc
    best_position: str
    best_layer: int
    best_accuracy: float
    probe_vs_dim_cosine: dict[str, dict[int, float]]
    exp1_cosine_agreement: dict[str, dict[int, float]] | None
    macro_axis_breakdown: dict[str, dict]            # axis -> metrics (at best pos+layer)
    ppl_stats: dict[str, float] = field(default_factory=dict)


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
    activation_cache_dir: Path | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
    max_response_chars: int = 3000,
    max_pairs: int | None = None,
    positions: tuple[str, ...] = DEFAULT_POSITIONS,
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

    # Extract activations + perplexity (one forward pass per batch).
    cache_root = Path(activation_cache_dir) if activation_cache_dir else out_dir
    cache_s = cache_root / "act_cache_sys_follow" if cache_activations else None
    cache_u = cache_root / "act_cache_usr_follow" if cache_activations else None

    with timer(f"[{model_key}] extract sys-following acts+ppl (bs={batch_size})"):
        sys_acts_full, sys_ppl = extract_multi_position_with_ppl_batched(
            loaded, sys_bundles,
            position_names=positions,
            batch_size=batch_size, max_length=max_length, cache_dir=cache_s,
        )
    with timer(f"[{model_key}] extract usr-following acts+ppl (bs={batch_size})"):
        usr_acts_full, usr_ppl = extract_multi_position_with_ppl_batched(
            loaded, usr_bundles,
            position_names=positions,
            batch_size=batch_size, max_length=max_length, cache_dir=cache_u,
        )
    # shape: (n, n_pos, n_layers, d)

    # Per-position probe training.
    from sklearn.metrics import accuracy_score, roc_auc_score

    all_bundles = sys_bundles + usr_bundles
    all_labels = np.array([1] * len(sys_bundles) + [0] * len(usr_bundles))
    all_ppl = np.concatenate([sys_ppl, usr_ppl], axis=0)

    probes_by_position: dict[str, dict] = {}
    accuracies_by_position: dict[str, dict[int, float]] = {}
    aurocs_by_position: dict[str, dict[int, float]] = {}
    directions_by_position: dict[str, dict[int, np.ndarray]] = {}
    dim_dirs_by_position: dict[str, dict[int, np.ndarray]] = {}
    agree_by_position: dict[str, dict[int, float]] = {}
    exp1_cos_by_position: dict[str, dict[int, float]] | None = None
    probe_result_by_position: dict[str, object] = {}

    exp1_dirs = _load_exp1_directions(Path(exp1_dir)) if exp1_dir else None
    if exp1_dirs is not None:
        exp1_cos_by_position = {}

    for pi, pos_name in enumerate(positions):
        sys_pos = sys_acts_full[:, pi]   # (n_sys, n_layers, d)
        usr_pos = usr_acts_full[:, pi]   # (n_usr, n_layers, d)
        with timer(f"[{model_key}] train probe @ {pos_name}"):
            pr = train_authority_probe(sys_pos, usr_pos, seed=seed)
        dim = compute_authority_direction_dim(sys_pos, usr_pos)
        agr = cosine_agreement(pr.directions, dim)

        accuracies_by_position[pos_name] = pr.accuracies
        aurocs_by_position[pos_name] = pr.aurocs
        directions_by_position[pos_name] = pr.directions
        dim_dirs_by_position[pos_name] = dim
        agree_by_position[pos_name] = agr
        probes_by_position[pos_name] = pr.probes
        probe_result_by_position[pos_name] = pr

        if exp1_dirs is not None:
            exp1_cos_by_position[pos_name] = cosine_agreement(pr.directions, exp1_dirs)

        best_l = find_best_layer(pr.accuracies)
        logger.info(
            f"[{model_key}] pos={pos_name:15s} best_layer={best_l:3d}"
            f" acc={pr.accuracies[best_l]:.4f} auc={pr.aurocs[best_l]:.4f}"
        )

    # Global best (position, layer).
    def _score(p, l):
        return accuracies_by_position[p][l]
    best_position, best_layer = max(
        ((p, l) for p in positions for l in range(loaded.n_layers)),
        key=lambda pl: _score(*pl),
    )
    best_accuracy = _score(best_position, best_layer)
    logger.info(
        f"[{model_key}] GLOBAL BEST: position={best_position} "
        f"layer={best_layer} acc={best_accuracy:.4f}"
    )

    # Per-macro-axis breakdown at global best (reuses that probe's test split).
    pr_best = probe_result_by_position[best_position]
    test_idx = pr_best.test_indices
    pos_index = positions.index(best_position)
    all_acts_best = np.concatenate(
        [sys_acts_full[:, pos_index], usr_acts_full[:, pos_index]], axis=0
    )
    X_test_best = all_acts_best[test_idx, best_layer, :]
    norms = np.linalg.norm(X_test_best, axis=1, keepdims=True) + 1e-8
    X_test_best = X_test_best / norms
    y_test = all_labels[test_idx]

    probe = probe_result_by_position[best_position].probes[best_layer]
    y_pred = probe.predict(X_test_best)
    y_prob = probe.predict_proba(X_test_best)[:, 1]

    axis_buckets: dict[str, dict] = {}
    for i_test, global_idx in enumerate(test_idx):
        axis = all_bundles[global_idx].extras.get("macro_axis", "unknown")
        axis_buckets.setdefault(axis, {"y_true": [], "y_pred": [], "y_prob": []})
        axis_buckets[axis]["y_true"].append(y_test[i_test])
        axis_buckets[axis]["y_pred"].append(y_pred[i_test])
        axis_buckets[axis]["y_prob"].append(y_prob[i_test])

    axis_results: dict[str, dict] = {}
    for axis, data in sorted(axis_buckets.items()):
        n_ax = len(data["y_true"])
        yt = np.array(data["y_true"]); yp = np.array(data["y_pred"]); yprob = np.array(data["y_prob"])
        acc = float(accuracy_score(yt, yp))
        try:
            auc = float(roc_auc_score(yt, yprob))
        except ValueError:
            auc = float("nan")
        axis_results[axis] = {"n": n_ax, "accuracy": acc, "auroc": auc}
        logger.info(f"  {axis:15s}  n={n_ax:4d}  acc={acc:.3f}  auc={auc:.3f}")

    # Per-prompt metadata CSV (for the Streamlit viewer).
    _write_prompts_csv(
        out_dir / "prompts.csv",
        sys_bundles, usr_bundles,
        sys_ppl, usr_ppl,
        sys_acts_full, usr_acts_full,
        directions_by_position,
        positions,
        best_position, best_layer,
    )

    # Persist arrays + probes.
    arrays: dict[str, np.ndarray] = {}
    for pos_name in positions:
        for l, v in directions_by_position[pos_name].items():
            arrays[f"probe_dir__{pos_name}__layer_{l:03d}"] = v.astype(np.float32)
        for l, v in dim_dirs_by_position[pos_name].items():
            arrays[f"dim_dir__{pos_name}__layer_{l:03d}"] = v.astype(np.float32)
        arrays[f"accuracies__{pos_name}"] = np.array(
            [accuracies_by_position[pos_name][i] for i in range(loaded.n_layers)],
            dtype=np.float32,
        )
        arrays[f"aurocs__{pos_name}"] = np.array(
            [aurocs_by_position[pos_name][i] for i in range(loaded.n_layers)],
            dtype=np.float32,
        )
    arrays["ppl_sys_following"] = sys_ppl.astype(np.float32)
    arrays["ppl_usr_following"] = usr_ppl.astype(np.float32)

    ppl_stats = {
        "sys_following_mean": float(np.nanmean(sys_ppl)),
        "sys_following_median": float(np.nanmedian(sys_ppl)),
        "usr_following_mean": float(np.nanmean(usr_ppl)),
        "usr_following_median": float(np.nanmedian(usr_ppl)),
    }

    result_json = {
        "model_key": model_key,
        "hf_id": loaded.cfg.hf_id,
        "n_layers": loaded.n_layers,
        "d_model": loaded.d_model,
        "n_pairs": len(pairs),
        "n_system_following": len(sys_bundles),
        "n_user_following": len(usr_bundles),
        "positions": list(positions),
        "accuracies_by_position": {
            p: {int(k): float(v) for k, v in accuracies_by_position[p].items()}
            for p in positions
        },
        "aurocs_by_position": {
            p: {int(k): float(v) for k, v in aurocs_by_position[p].items()}
            for p in positions
        },
        "probe_vs_dim_cosine_by_position": {
            p: {int(k): float(v) for k, v in agree_by_position[p].items()}
            for p in positions
        },
        "exp1_cosine_agreement_by_position": (
            {p: {int(k): float(v) for k, v in exp1_cos_by_position[p].items()}
             for p in positions}
            if exp1_cos_by_position is not None else None
        ),
        "best_position": best_position,
        "best_layer": int(best_layer),
        "best_accuracy": float(best_accuracy),
        "macro_axis_breakdown": axis_results,
        "ppl_stats": ppl_stats,
        "test_idx_by_position": {
            p: probe_result_by_position[p].test_indices.tolist() for p in positions
        },
    }

    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        pickles={"probes_by_position": probes_by_position},
        manifest_extras={
            "experiment": "exp1b_authority_conflict",
            "model_key": model_key,
            "positions": list(positions),
        },
    )
    save_pickle(probe_result_by_position, out_dir / "probe_result_by_position.pkl")

    if free_after:
        free_model(loaded)

    return Exp1bResult(
        model_key=model_key,
        n_layers=loaded.n_layers,
        d_model=loaded.d_model,
        n_pairs=len(pairs),
        n_system_following=len(sys_bundles),
        n_user_following=len(usr_bundles),
        positions=tuple(positions),
        accuracies=accuracies_by_position,
        aurocs=aurocs_by_position,
        best_position=best_position,
        best_layer=best_layer,
        best_accuracy=best_accuracy,
        probe_vs_dim_cosine=agree_by_position,
        exp1_cosine_agreement=exp1_cos_by_position,
        macro_axis_breakdown=axis_results,
        ppl_stats=ppl_stats,
    )


def _write_prompts_csv(
    path: Path,
    sys_bundles,
    usr_bundles,
    sys_ppl: np.ndarray,
    usr_ppl: np.ndarray,
    sys_acts_full: np.ndarray,
    usr_acts_full: np.ndarray,
    directions_by_position: dict[str, dict[int, np.ndarray]],
    positions: tuple[str, ...],
    best_position: str,
    best_layer: int,
) -> None:
    """Dump one row per trace: pair_idx, trace, label, axis, ppl, probe score at best (pos, layer)."""
    pos_index = positions.index(best_position)
    best_dir = directions_by_position[best_position][best_layer]
    best_dir = best_dir / (np.linalg.norm(best_dir) + 1e-8)

    def _score(acts_full: np.ndarray, side_idx: int) -> float:
        v = acts_full[side_idx, pos_index, best_layer]
        vn = v / (np.linalg.norm(v) + 1e-8)
        return float(vn @ best_dir)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "side", "idx", "pair_idx", "trace", "label", "macro_axis",
            "conflict_axis", "ppl", "probe_score_best",
            "response_preview",
        ])
        for i, b in enumerate(sys_bundles):
            ex = b.extras or {}
            w.writerow([
                "sys_following", i, ex.get("pair_idx"), ex.get("trace"),
                1, ex.get("macro_axis", "unknown"), ex.get("conflict_axis", ""),
                f"{float(sys_ppl[i]):.6f}" if not np.isnan(sys_ppl[i]) else "",
                f"{_score(sys_acts_full, i):.6f}",
                (b.instruction_text or "")[:200],
            ])
        for i, b in enumerate(usr_bundles):
            ex = b.extras or {}
            w.writerow([
                "usr_following", i, ex.get("pair_idx"), ex.get("trace"),
                0, ex.get("macro_axis", "unknown"), ex.get("conflict_axis", ""),
                f"{float(usr_ppl[i]):.6f}" if not np.isnan(usr_ppl[i]) else "",
                f"{_score(usr_acts_full, i):.6f}",
                (b.instruction_text or "")[:200],
            ])
