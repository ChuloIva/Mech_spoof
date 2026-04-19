"""Experiment 2 — Conflict behavioral test (§5).

For each conflict pair, generate responses under REAL / NONE / FAKE conditions, evaluate whether
the model followed the system instruction, and record the probe score at response_first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

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


def run_experiment_2(
    model_key: str,
    out_dir: Path,
    exp1_dir: Optional[Path] = None,
    max_new_tokens: int = 1024,
    seed: int = 42,
    batch_size: int = 8,
    free_after: bool = True,
) -> Exp2Result:
    """Run Experiment 2 end-to-end for one model.

    Batched generate with prefill-only residual capture via a forward hook on the target
    block. The hook fires on every forward pass but only stores activations on the first
    call per batch (the prefill) — avoiding the memory cost of output_hidden_states=True
    and the wall-time cost of a second forward pass per prompt.
    """
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
    direction_norm = None
    if direction is not None:
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

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

    # Hook captures only the first forward pass per batch (the prefill).
    prefill_state = {"last_token_acts": None, "captured": False}
    hook_handle = None
    if direction_norm is not None:
        target_block = loaded.layer_module(int(best_layer))

        def _prefill_hook(_m, _inp, out, state=prefill_state):
            if state["captured"]:
                return
            h = out[0] if isinstance(out, tuple) else out
            state["last_token_acts"] = h[:, -1, :].detach().float().cpu().numpy()
            state["captured"] = True

        hook_handle = target_block.register_forward_hook(_prefill_hook)

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

            prefill_state["last_token_acts"] = None
            prefill_state["captured"] = False

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

            probe_scores: list[float | None] = [None] * len(chunk)
            if direction_norm is not None and prefill_state["last_token_acts"] is not None:
                acts = prefill_state["last_token_acts"]  # (batch, d)
                for j in range(len(chunk)):
                    vec = acts[j]
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    probe_scores[j] = float(vec @ direction_norm)

            for j, (cp, cond, _bundle) in enumerate(chunk):
                response = tok.decode(gen_tokens[j], skip_special_tokens=True)
                system_followed = evaluate_compliance(response, cp.pair, which="system")
                rows.append({
                    "pair_id": cp.pair.id,
                    "category": cp.pair.category,
                    "eval_type": cp.pair.eval,
                    "condition": cond,
                    "system_followed": None if system_followed is None else bool(system_followed),
                    "probe_score": probe_scores[j],
                    "response": response,
                })

            del gen_out, input_ids, attention_mask, enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        tok.padding_side = original_padding

    # Aggregate. Rows with system_followed=None are unjudgable (judge parse failed)
    # and are excluded from both compliance_rate and correlation.
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
            "n_judged": len(judged),
            "n_unjudgable": len(subset) - len(judged),
            "mean_probe_score": float(np.mean(scores)) if scores else None,
            "std_probe_score": float(np.std(scores)) if scores else None,
            "n": len(subset),
        }

    correlation: dict = {}
    if direction is not None:
        corr_rows = [
            r for r in rows
            if r["probe_score"] is not None and r["system_followed"] is not None
        ]
        all_scores = np.array([r["probe_score"] for r in corr_rows])
        all_labels = np.array([1 if r["system_followed"] else 0 for r in corr_rows])
        if len(all_scores) > 2 and len(np.unique(all_labels)) > 1:
            r_val, p_val = pearsonr(all_scores, all_labels)
            correlation["overall"] = {"r": float(r_val), "p": float(p_val), "n": int(len(all_scores))}
        for cond in ("REAL", "NONE", "FAKE"):
            subset = [
                r for r in rows
                if r["condition"] == cond
                and r["probe_score"] is not None
                and r["system_followed"] is not None
            ]
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
