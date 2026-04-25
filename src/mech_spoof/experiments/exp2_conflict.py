"""Experiment 2 — Conflict behavioral test (§5).

For each conflict pair, generate responses under REAL / NONE / FAKE conditions, evaluate whether
the model followed the system instruction, and record probe scores at response_first for every
layer (not just the best). Per-row we also count how many layers "activate" — i.e. land above
the layer's own median projection — as a coarse proxy for how many depths carry the authority
signal on that prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from mech_spoof.datasets.conflicting import build_conflicting_pairs
from mech_spoof.eval.compliance import evaluate_compliance
from mech_spoof.io import load_authority_directions, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp2Result:
    model_key: str
    n_pairs: int
    summary: dict          # per-condition compliance rate + best-layer probe stats
    correlation: dict      # best-layer per-condition + per-layer + n-activated


def run_experiment_2(
    model_key: str,
    out_dir: Path,
    exp1_dir: Optional[Path] = None,
    probe_position: str | None = None,
    max_new_tokens: int = 1024,
    seed: int = 42,
    batch_size: int = 8,
    free_after: bool = True,
) -> Exp2Result:
    """Run Experiment 2 end-to-end for one model.

    Batched generate with prefill-only residual capture via forward hooks on every target
    block. Each hook fires on every forward pass but only stores activations on the first
    call per batch (the prefill) — avoiding the memory cost of output_hidden_states=True
    and the wall-time cost of a second forward pass per prompt.

    Per-layer probe projections are recorded for every row so we can see which depths
    (if any) actually carry a behaviorally-predictive authority signal.
    """
    import torch
    from scipy.stats import pearsonr

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(model_key)

    # Load authority probe directions for every layer (optional).
    loaded_dirs = (
        load_authority_directions(exp1_dir, position=probe_position)
        if exp1_dir else None
    )
    if loaded_dirs is None:
        logger.warning(
            f"[{model_key}] exp1/exp1b bundle not found — skipping probe-score correlation"
        )
        best_layer: int | None = None
        direction_norms: dict[int, np.ndarray] = {}
        resolved_position: str | None = None
    else:
        best_layer, raw_dirs, resolved_position = loaded_dirs
        direction_norms = {
            l: (v / (np.linalg.norm(v) + 1e-8)) for l, v in raw_dirs.items()
        }
        pos_msg = f" position={resolved_position}" if resolved_position else ""
        logger.info(
            f"[{model_key}] loaded {len(direction_norms)} probe directions;"
            f" best_layer={best_layer}{pos_msg}"
        )

    conflict_prompts = build_conflicting_pairs(loaded.template)
    logger.info(f"[{model_key}] conflict pairs: {len(conflict_prompts)}")

    from tqdm.auto import tqdm

    # Flatten (pair, condition, bundle) triples so we can batch across conditions.
    flat: list[tuple] = []
    for cp in conflict_prompts:
        flat.append((cp, "REAL", cp.real))
        flat.append((cp, "NONE", cp.none))
        flat.append((cp, "FAKE", cp.fake))

    tok = loaded.tokenizer
    original_padding = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Per-layer prefill capture. Each hook projects its layer's last-token activation
    # onto that layer's probe direction and stashes a (batch,)-shaped array of scalars.
    layers_tracked: list[int] = sorted(direction_norms.keys())
    prefill_scores: dict[int, np.ndarray | None] = {l: None for l in layers_tracked}
    captured: dict[int, bool] = {l: False for l in layers_tracked}
    hook_handles: list = []

    def _make_hook(layer_idx: int):
        dnorm = direction_norms[layer_idx]

        def _hook(_m, _inp, out):
            if captured[layer_idx]:
                return
            h = out[0] if isinstance(out, tuple) else out
            vec = h[:, -1, :].detach().float().cpu().numpy()  # (batch, d)
            norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
            prefill_scores[layer_idx] = (vec / norms) @ dnorm  # (batch,)
            captured[layer_idx] = True

        return _hook

    for l in layers_tracked:
        hook_handles.append(loaded.layer_module(l).register_forward_hook(_make_hook(l)))

    rows: list[dict] = []
    try:
        iterator = range(0, len(flat), batch_size)
        for start in tqdm(iterator, desc=f"exp2 {model_key} bs={batch_size}"):
            chunk = flat[start:start + batch_size]
            texts = [b.text for (_, _, b) in chunk]

            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(loaded.device)
            attention_mask = enc["attention_mask"].to(loaded.device)

            for l in layers_tracked:
                prefill_scores[l] = None
                captured[l] = False

            with torch.no_grad():
                gen_out = loaded.hf_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )

            prompt_len = input_ids.shape[1]
            gen_tokens = gen_out[:, prompt_len:]

            # Assemble per-row, per-layer probe scores from the prefill capture.
            probe_scores_by_layer: list[dict[int, float]] = [dict() for _ in chunk]
            for l in layers_tracked:
                batch_scores = prefill_scores[l]
                if batch_scores is None:
                    continue
                for j in range(len(chunk)):
                    probe_scores_by_layer[j][l] = float(batch_scores[j])

            # Backward-compat scalar at the best layer.
            best_scores: list[float | None] = [None] * len(chunk)
            if best_layer is not None:
                for j in range(len(chunk)):
                    best_scores[j] = probe_scores_by_layer[j].get(int(best_layer))

            for j, (cp, cond, _bundle) in enumerate(chunk):
                response = tok.decode(gen_tokens[j], skip_special_tokens=True)
                system_followed = evaluate_compliance(response, cp.pair, which="system")
                rows.append({
                    "pair_id": cp.pair.id,
                    "category": cp.pair.category,
                    "eval_type": cp.pair.eval,
                    "condition": cond,
                    "system_followed": None if system_followed is None else bool(system_followed),
                    "probe_score": best_scores[j],
                    "probe_scores_by_layer": probe_scores_by_layer[j] or None,
                    "response": response,
                })

            del gen_out, input_ids, attention_mask, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hook_handles:
            h.remove()
        tok.padding_side = original_padding

    # ---- Derived per-row features: z-scored per-layer scores + activation counts ----
    # A layer is "activated" for a row if its probe projection on that row lands above
    # the layer's own population median (z > 0). The count across layers is a coarse
    # proxy for how widely the authority signal fires in the residual stream on that
    # prompt. We also keep a stricter "strong" count (z > 1).
    if layers_tracked:
        score_matrix = np.full((len(rows), len(layers_tracked)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            sbl = r.get("probe_scores_by_layer") or {}
            for k, l in enumerate(layers_tracked):
                v = sbl.get(l)
                if v is not None:
                    score_matrix[i, k] = v
        # Per-layer standardization across all rows (NaN-safe; layers with all-NaN stay NaN).
        layer_means = np.nanmean(score_matrix, axis=0, keepdims=True)
        layer_stds = np.nanstd(score_matrix, axis=0, keepdims=True)
        layer_stds = np.where(layer_stds < 1e-12, 1.0, layer_stds)
        zs = (score_matrix - layer_means) / layer_stds

        above_mean = (zs > 0).astype(np.int32)  # includes NaN -> False
        strong = (zs > 1.0).astype(np.int32)
        n_above_mean = above_mean.sum(axis=1)
        n_strong = strong.sum(axis=1)

        for i, r in enumerate(rows):
            r["n_layers_activated_above_mean"] = int(n_above_mean[i])
            r["n_layers_activated_strong"] = int(n_strong[i])
    else:
        score_matrix = None
        zs = None

    # ---- Aggregate. Rows with system_followed=None are unjudgable and are dropped. ----
    summary: dict = {}
    for cond in ("REAL", "NONE", "FAKE"):
        subset = [r for r in rows if r["condition"] == cond]
        judged = [r for r in subset if r["system_followed"] is not None]
        comply = (
            float(np.mean([1 if r["system_followed"] else 0 for r in judged]))
            if judged else None
        )
        scores = [r["probe_score"] for r in subset if r["probe_score"] is not None]
        act_counts = [
            r.get("n_layers_activated_above_mean") for r in subset
            if r.get("n_layers_activated_above_mean") is not None
        ]
        strong_counts = [
            r.get("n_layers_activated_strong") for r in subset
            if r.get("n_layers_activated_strong") is not None
        ]
        summary[cond] = {
            "compliance_rate": comply,
            "n_judged": len(judged),
            "n_unjudgable": len(subset) - len(judged),
            "mean_probe_score": float(np.mean(scores)) if scores else None,
            "std_probe_score": float(np.std(scores)) if scores else None,
            "mean_n_activated_above_mean": float(np.mean(act_counts)) if act_counts else None,
            "mean_n_activated_strong": float(np.mean(strong_counts)) if strong_counts else None,
            "n": len(subset),
        }

    # ---- Correlations ----
    correlation: dict = {}

    def _pearson_or_none(x: np.ndarray, y: np.ndarray) -> dict | None:
        if len(x) < 3 or len(np.unique(y)) < 2:
            return None
        r_val, p_val = pearsonr(x, y)
        return {"r": float(r_val), "p": float(p_val), "n": int(len(x))}

    if layers_tracked:
        # Best-layer correlations (back-compat with earlier analysis).
        corr_rows = [
            r for r in rows
            if r["probe_score"] is not None and r["system_followed"] is not None
        ]
        if corr_rows:
            xs = np.array([r["probe_score"] for r in corr_rows])
            ys = np.array([1 if r["system_followed"] else 0 for r in corr_rows])
            res = _pearson_or_none(xs, ys)
            if res is not None:
                correlation["overall"] = res
            for cond in ("REAL", "NONE", "FAKE"):
                subset = [r for r in corr_rows if r["condition"] == cond]
                if len(subset) >= 3:
                    xs_c = np.array([r["probe_score"] for r in subset])
                    ys_c = np.array([1 if r["system_followed"] else 0 for r in subset])
                    res_c = _pearson_or_none(xs_c, ys_c)
                    if res_c is not None:
                        correlation[cond] = res_c

        # Per-layer correlations (overall + per condition).
        by_layer: dict[int, dict] = {}
        assert score_matrix is not None  # layers_tracked nonempty ⇒ matrix exists
        row_followed = np.array([
            (1 if r["system_followed"] else 0) if r["system_followed"] is not None else -1
            for r in rows
        ])
        row_cond = np.array([r["condition"] for r in rows])
        valid_mask = row_followed >= 0  # exclude unjudgable rows

        for k, l in enumerate(layers_tracked):
            col = score_matrix[:, k]
            finite_mask = np.isfinite(col) & valid_mask
            entry: dict = {}
            if finite_mask.sum() >= 3:
                res = _pearson_or_none(col[finite_mask], row_followed[finite_mask])
                if res is not None:
                    entry["overall"] = res
            for cond in ("REAL", "NONE", "FAKE"):
                cond_mask = finite_mask & (row_cond == cond)
                if cond_mask.sum() >= 3:
                    res_c = _pearson_or_none(col[cond_mask], row_followed[cond_mask])
                    if res_c is not None:
                        entry[cond] = res_c
            if entry:
                by_layer[int(l)] = entry
        if by_layer:
            correlation["by_layer"] = by_layer

        # n-layers-activated correlations.
        for field, label in (
            ("n_layers_activated_above_mean", "n_activated_above_mean"),
            ("n_layers_activated_strong", "n_activated_strong"),
        ):
            nrows = [
                r for r in rows
                if r.get(field) is not None and r["system_followed"] is not None
            ]
            entry = {}
            if len(nrows) >= 3:
                xs = np.array([r[field] for r in nrows], dtype=float)
                ys = np.array([1 if r["system_followed"] else 0 for r in nrows])
                res = _pearson_or_none(xs, ys)
                if res is not None:
                    entry["overall"] = res
                for cond in ("REAL", "NONE", "FAKE"):
                    sub = [r for r in nrows if r["condition"] == cond]
                    if len(sub) >= 3:
                        xs_c = np.array([r[field] for r in sub], dtype=float)
                        ys_c = np.array([1 if r["system_followed"] else 0 for r in sub])
                        res_c = _pearson_or_none(xs_c, ys_c)
                        if res_c is not None:
                            entry[cond] = res_c
            if entry:
                correlation[label] = entry

    result_json = {
        "model_key": model_key,
        "n_pairs": len(conflict_prompts),
        "exp1_best_layer": int(best_layer) if best_layer is not None else None,
        "probe_position": resolved_position,
        "layers_tracked": layers_tracked,
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
