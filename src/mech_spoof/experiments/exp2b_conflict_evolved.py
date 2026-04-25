"""Experiment 2b — Behavioral test on held-out evolved conflict pairs.

Same residual-capture and per-layer correlation pipeline as exp2, but uses pairs sampled
from the held-out portion of the UltraFeedback-evolved conflict dataset (the ~20k pairs
that were NOT in the stratified 1k used to train the exp1b probe). This is the
in-distribution generalization test for the probe.

Compliance is judged by a vLLM-served instruct model (free-form responses → an LLM grader
with the gold S/U responses as references). HF model is freed before vLLM is loaded
to avoid VRAM contention.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mech_spoof.datasets.conflict_evolved import (
    EvolvedConflictPair,
    load_held_out_evolved_pairs,
)
from mech_spoof.eval.llm_judge_vllm import JudgeRow, judge_with_vllm
from mech_spoof.io import load_authority_directions, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp2bResult:
    model_key: str
    n_pairs: int
    n_judged: int
    summary: dict
    correlation: dict
    judge_model_id: str
    probe_position: str | None


def _build_conditions(template, pair: EvolvedConflictPair):
    """Return (REAL, NONE, FAKE) PromptBundles for one pair."""
    return {
        "REAL": template.make_conflict_prompt(pair.s_instruction, pair.u_instruction, "REAL"),
        "NONE": template.make_conflict_prompt(pair.s_instruction, pair.u_instruction, "NONE"),
        "FAKE": template.make_conflict_prompt(pair.s_instruction, pair.u_instruction, "FAKE"),
    }


def run_experiment_2b(
    model_key: str,
    out_dir: Path,
    exp1_dir: Path,
    probe_position: str | None = None,
    judge_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    n_pairs: int = 200,
    max_new_tokens: int = 512,
    seed: int = 42,
    batch_size: int = 8,
    judge_batch_size: int = 64,
    judge_max_model_len: int = 8192,
    judge_gpu_memory_utilization: float = 0.85,
    free_after: bool = True,
) -> Exp2bResult:
    """Run Exp 2b end-to-end. Two phases: (1) HF generate + per-layer probe scores via
    forward hooks, (2) free HF, load vLLM judge, score compliance. The split is
    important because both models would otherwise contend for VRAM."""
    import torch
    from scipy.stats import pearsonr
    from tqdm.auto import tqdm

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- 0. Load probe directions -----
    loaded_dirs = load_authority_directions(exp1_dir, position=probe_position)
    if loaded_dirs is None:
        raise FileNotFoundError(
            f"Exp2b requires an authority probe bundle at {exp1_dir} (exp1 or exp1b)."
        )
    best_layer, raw_dirs, resolved_position = loaded_dirs
    direction_norms = {
        l: (v / (np.linalg.norm(v) + 1e-8)) for l, v in raw_dirs.items()
    }
    logger.info(
        f"[{model_key}] probe: best_layer={best_layer}"
        f" position={resolved_position} n_dirs={len(direction_norms)}"
    )

    # ----- 1. Held-out pairs -----
    pairs = load_held_out_evolved_pairs(n_held_out=n_pairs, seed=seed)
    logger.info(f"[{model_key}] held-out pairs: {len(pairs)}")

    # ----- 2. Build prompt bundles, flatten across conditions -----
    loaded = load_model(model_key)
    template = loaded.template
    flat: list[tuple[EvolvedConflictPair, str, object]] = []
    for p in pairs:
        bundles = _build_conditions(template, p)
        for cond, b in bundles.items():
            flat.append((p, cond, b))

    tok = loaded.tokenizer
    original_padding = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # ----- 3. Per-layer prefill capture hooks (same trick as exp2) -----
    layers_tracked = sorted(direction_norms.keys())
    prefill_scores: dict[int, np.ndarray | None] = {l: None for l in layers_tracked}
    captured: dict[int, bool] = {l: False for l in layers_tracked}
    hook_handles: list = []

    def _make_hook(layer_idx: int):
        dnorm = direction_norms[layer_idx]
        def _hook(_m, _inp, out):
            if captured[layer_idx]:
                return
            h = out[0] if isinstance(out, tuple) else out
            vec = h[:, -1, :].detach().float().cpu().numpy()
            norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
            prefill_scores[layer_idx] = (vec / norms) @ dnorm
            captured[layer_idx] = True
        return _hook

    for l in layers_tracked:
        hook_handles.append(loaded.layer_module(l).register_forward_hook(_make_hook(l)))

    # ----- 4. Generate + collect probe scores -----
    rows: list[dict] = []
    try:
        iterator = range(0, len(flat), batch_size)
        for start in tqdm(iterator, desc=f"exp2b {model_key} bs={batch_size}"):
            chunk = flat[start: start + batch_size]
            texts = [b.text for (_, _, b) in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
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

            probe_scores_by_layer: list[dict[int, float]] = [dict() for _ in chunk]
            for l in layers_tracked:
                bs_scores = prefill_scores[l]
                if bs_scores is None:
                    continue
                for j in range(len(chunk)):
                    probe_scores_by_layer[j][l] = float(bs_scores[j])

            best_scores: list[float | None] = [
                probe_scores_by_layer[j].get(int(best_layer))
                for j in range(len(chunk))
            ]

            for j, (pair, cond, _b) in enumerate(chunk):
                response = tok.decode(gen_tokens[j], skip_special_tokens=True)
                rows.append({
                    "pair_idx": pair.idx,
                    "conflict_axis": pair.conflict_axis,
                    "condition": cond,
                    "s_instruction": pair.s_instruction,
                    "u_instruction": pair.u_instruction,
                    "s_gold": pair.s_aligned_response,
                    "u_gold": pair.u_aligned_response,
                    "response": response,
                    "probe_score": best_scores[j],
                    "probe_scores_by_layer": probe_scores_by_layer[j] or None,
                })

            del gen_out, input_ids, attention_mask, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for h in hook_handles:
            h.remove()
        tok.padding_side = original_padding

    # ----- 5. Free HF model BEFORE loading vLLM judge -----
    free_model(loaded)
    del loaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ----- 6. vLLM judge -----
    judge_inputs = [
        JudgeRow(
            s_instruction=r["s_instruction"],
            u_instruction=r["u_instruction"],
            s_gold=r["s_gold"],
            u_gold=r["u_gold"],
            response=r["response"],
            extras={"row_idx": i, "condition": r["condition"], "pair_idx": r["pair_idx"]},
        )
        for i, r in enumerate(rows)
    ]
    with timer(f"[{model_key}] vLLM judge ({judge_model_id})"):
        verdicts = judge_with_vllm(
            judge_inputs,
            model_id=judge_model_id,
            max_model_len=judge_max_model_len,
            gpu_memory_utilization=judge_gpu_memory_utilization,
            batch_size=judge_batch_size,
            seed=seed,
            free_after=True,
        )

    for r, v in zip(rows, verdicts):
        verdict = v["verdict"]
        r["judge_verdict"] = verdict
        r["judge_reason"] = v["reason"]
        if verdict == "system":
            r["system_followed"] = True
        elif verdict == "user":
            r["system_followed"] = False
        else:
            r["system_followed"] = None

    # ----- 7. Per-layer activation counts (z>0, z>1) -----
    score_matrix = None
    if layers_tracked:
        score_matrix = np.full((len(rows), len(layers_tracked)), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            sbl = r.get("probe_scores_by_layer") or {}
            for k, l in enumerate(layers_tracked):
                v = sbl.get(l)
                if v is not None:
                    score_matrix[i, k] = v
        layer_means = np.nanmean(score_matrix, axis=0, keepdims=True)
        layer_stds = np.nanstd(score_matrix, axis=0, keepdims=True)
        layer_stds = np.where(layer_stds < 1e-12, 1.0, layer_stds)
        zs = (score_matrix - layer_means) / layer_stds
        n_above_mean = (zs > 0).astype(np.int32).sum(axis=1)
        n_strong = (zs > 1.0).astype(np.int32).sum(axis=1)
        for i, r in enumerate(rows):
            r["n_layers_activated_above_mean"] = int(n_above_mean[i])
            r["n_layers_activated_strong"] = int(n_strong[i])

    # ----- 8. Aggregate -----
    summary: dict = {}
    for cond in ("REAL", "NONE", "FAKE"):
        subset = [r for r in rows if r["condition"] == cond]
        judged = [r for r in subset if r["system_followed"] is not None]
        comply = (
            float(np.mean([1 if r["system_followed"] else 0 for r in judged]))
            if judged else None
        )
        scores = [r["probe_score"] for r in subset if r["probe_score"] is not None]
        summary[cond] = {
            "compliance_rate": comply,
            "n": len(subset),
            "n_judged": len(judged),
            "n_unjudgable": len(subset) - len(judged),
            "mean_probe_score": float(np.mean(scores)) if scores else None,
            "std_probe_score": float(np.std(scores)) if scores else None,
        }

    correlation: dict = {}

    def _pearson_or_none(x: np.ndarray, y: np.ndarray) -> dict | None:
        if len(x) < 3 or len(np.unique(y)) < 2:
            return None
        r_val, p_val = pearsonr(x, y)
        return {"r": float(r_val), "p": float(p_val), "n": int(len(x))}

    if layers_tracked:
        # Best-layer correlations.
        corr_rows = [
            r for r in rows
            if r["probe_score"] is not None and r["system_followed"] is not None
        ]
        if corr_rows:
            xs = np.array([r["probe_score"] for r in corr_rows])
            ys = np.array([1 if r["system_followed"] else 0 for r in corr_rows])
            r_overall = _pearson_or_none(xs, ys)
            if r_overall is not None:
                correlation["overall"] = r_overall
            for cond in ("REAL", "NONE", "FAKE"):
                sub = [r for r in corr_rows if r["condition"] == cond]
                if len(sub) >= 3:
                    xs_c = np.array([r["probe_score"] for r in sub])
                    ys_c = np.array([1 if r["system_followed"] else 0 for r in sub])
                    res = _pearson_or_none(xs_c, ys_c)
                    if res is not None:
                        correlation[cond] = res

        # Per-layer correlations.
        by_layer: dict[int, dict] = {}
        row_followed = np.array([
            (1 if r["system_followed"] else 0) if r["system_followed"] is not None else -1
            for r in rows
        ])
        row_cond = np.array([r["condition"] for r in rows])
        valid_mask = row_followed >= 0
        for k, l in enumerate(layers_tracked):
            col = score_matrix[:, k]
            finite_mask = np.isfinite(col) & valid_mask
            entry: dict = {}
            if finite_mask.sum() >= 3:
                res = _pearson_or_none(col[finite_mask], row_followed[finite_mask])
                if res is not None:
                    entry["overall"] = res
            for cond in ("REAL", "NONE", "FAKE"):
                cm = finite_mask & (row_cond == cond)
                if cm.sum() >= 3:
                    res = _pearson_or_none(col[cm], row_followed[cm])
                    if res is not None:
                        entry[cond] = res
            if entry:
                by_layer[int(l)] = entry
        if by_layer:
            correlation["by_layer"] = by_layer

    # ----- 9. Save -----
    n_judged_total = sum(1 for r in rows if r["system_followed"] is not None)
    result_json = {
        "model_key": model_key,
        "n_pairs": len(pairs),
        "n_rows": len(rows),
        "n_judged": n_judged_total,
        "exp1_best_layer": int(best_layer) if best_layer is not None else None,
        "probe_position": resolved_position,
        "judge_model_id": judge_model_id,
        "layers_tracked": layers_tracked,
        "summary": summary,
        "correlation": correlation,
        "rows": rows,
    }
    save_result_bundle(
        out_dir,
        json_obj=result_json,
        manifest_extras={"experiment": "exp2b_conflict_evolved", "model_key": model_key},
    )

    return Exp2bResult(
        model_key=model_key,
        n_pairs=len(pairs),
        n_judged=n_judged_total,
        summary=summary,
        correlation=correlation,
        judge_model_id=judge_model_id,
        probe_position=resolved_position,
    )
