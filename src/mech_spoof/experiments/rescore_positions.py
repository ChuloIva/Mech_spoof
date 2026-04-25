"""Rescore exp2b's saved generations using exp1b probes at *all* extraction positions.

Exp2b captures activations at the prefill end (last input token, before generation). Exp1b
trained probes at three positions: response_first, response_mid, response_last. The default
exp2b run only used one (whichever the bundle marked as `best_position`). This script rebuilds
the full (prompt + response) sequence per row, runs one extra forward pass, captures
activations at all three positions across all layers, and projects each onto the matching
probe direction set.

Output: a CSV with per-position best-layer probe scores + per-layer score JSON, optionally
joined with judge verdicts from a judged_*.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from mech_spoof.activations import extract_multi_position_with_ppl_batched
from mech_spoof.io import load_json, load_npz
from mech_spoof.models import free_model, load_model
from mech_spoof.templates.base import PromptBundle
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)

DEFAULT_POSITIONS: tuple[str, ...] = ("response_first", "response_mid", "response_last")


def _flatten_ids(ids):
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _build_bundle(
    tok,
    template,
    s_instruction: str,
    u_instruction: str,
    condition: str,
    response: str,
    supports_thinking: bool,
    system_role_supported: bool,
) -> PromptBundle:
    """Reconstruct a bundle for [prompt(condition) + response] with response_token_span set.

    Mirrors how exp2b built each prompt for generation, then appends the saved response
    as the assistant turn so we can capture residuals at response_first / mid / last.
    """
    if condition == "REAL":
        if system_role_supported:
            messages_full = [
                {"role": "system", "content": s_instruction},
                {"role": "user", "content": u_instruction},
                {"role": "assistant", "content": response},
            ]
        else:
            messages_full = [
                {"role": "user", "content": f"{s_instruction}\n\n{u_instruction}"},
                {"role": "assistant", "content": response},
            ]
    elif condition == "NONE":
        merged = f"{s_instruction}\n\n{u_instruction}"
        messages_full = [
            {"role": "user", "content": merged},
            {"role": "assistant", "content": response},
        ]
    elif condition == "NONE_REV":
        merged = f"{u_instruction}\n\n{s_instruction}"
        messages_full = [
            {"role": "user", "content": merged},
            {"role": "assistant", "content": response},
        ]
    elif condition == "FAKE":
        fake = template.build_fake_delimiter_injection(s_instruction)
        messages_full = [
            {"role": "user", "content": f"{fake}\n\n{u_instruction}"},
            {"role": "assistant", "content": response},
        ]
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    extra_kwargs = {"enable_thinking": False} if supports_thinking else {}
    text = tok.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False, **extra_kwargs
    )
    full_ids = _flatten_ids(tok.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False, **extra_kwargs
    ))
    prefix_ids = _flatten_ids(tok.apply_chat_template(
        messages_full[:-1], tokenize=True, add_generation_prompt=True, **extra_kwargs
    ))

    response_start = min(len(prefix_ids), len(full_ids))
    response_end = len(full_ids)
    if response_end <= response_start:
        response_start = max(0, response_end - 1)

    return PromptBundle(
        text=text,
        input_ids=full_ids,
        instruction_text=response[:100],
        instruction_token_span=(0, response_start),
        response_first_pos=response_start,
        condition=condition,
        extras={"response_token_span": (response_start, response_end)},
    )


def _load_probe_dirs_all_positions(
    exp1_dir: Path,
    positions: tuple[str, ...],
) -> tuple[dict[str, dict[int, np.ndarray]], int, str | None]:
    """Load probe directions for every position from an exp1b bundle.

    Returns (dirs_by_position, best_layer, best_position_or_None).
    Each direction is L2-normalized and float32.
    """
    arrays = load_npz(exp1_dir / "arrays.npz")
    meta = load_json(exp1_dir / "result.json")
    best_layer = int(meta["best_layer"])
    best_position = meta.get("best_position")

    dirs_by_position: dict[str, dict[int, np.ndarray]] = {p: {} for p in positions}
    for key, val in arrays.items():
        if not key.startswith("probe_dir__"):
            continue
        # probe_dir__<position>__layer_NNN
        parts = key.split("__")
        if len(parts) != 3:
            continue
        _, pos_name, layer_part = parts
        if pos_name not in dirs_by_position:
            continue
        try:
            layer = int(layer_part.replace("layer_", ""))
        except ValueError:
            continue
        norm = float(np.linalg.norm(val))
        if norm < 1e-8:
            continue
        dirs_by_position[pos_name][layer] = (val / norm).astype(np.float32)

    for p in positions:
        if not dirs_by_position[p]:
            raise KeyError(f"No probe directions found for position '{p}' in {exp1_dir}")
        logger.info(f"position {p}: loaded {len(dirs_by_position[p])} probe directions")

    return dirs_by_position, best_layer, best_position


def _read_judge_csv(path: Path) -> dict[tuple[str, str], dict]:
    """Index a judged_*.csv by (pair_idx, condition) -> row dict."""
    out: dict[tuple[str, str], dict] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (str(row.get("pair_idx", "")), str(row.get("condition", "")))
            out[key] = row
    return out


def rescore_exp2b_at_all_positions(
    model_key: str,
    exp1_dir: Path,
    exp2b_result_path: Path,
    out_csv_path: Path,
    judge_csv_path: Path | None = None,
    batch_size: int = 4,
    max_length: int = 4096,
    max_response_chars: int = 4000,
    seed: int = 42,
    positions: tuple[str, ...] = DEFAULT_POSITIONS,
    free_after: bool = True,
) -> dict:
    """End-to-end rescoring. Returns a summary dict with per-position correlation stats."""
    set_seed(seed)
    exp1_dir = Path(exp1_dir)
    exp2b_result_path = Path(exp2b_result_path)
    out_csv_path = Path(out_csv_path)
    judge_csv_path = Path(judge_csv_path) if judge_csv_path else None

    dirs_by_position, best_layer, best_position = _load_probe_dirs_all_positions(
        exp1_dir, positions
    )
    logger.info(
        f"probe meta: best_layer={best_layer}  best_position={best_position}"
    )

    exp2b = load_json(exp2b_result_path)
    rows = exp2b["rows"]
    logger.info(f"Loaded {len(rows)} rows from {exp2b_result_path}")

    loaded = load_model(model_key)
    template = loaded.template
    supports_thinking = getattr(template, "_supports_enable_thinking", False)
    system_role_supported = getattr(template, "_system_role_supported", True)

    bundles: list[PromptBundle] = []
    for r in rows:
        resp = r.get("response", "") or ""
        if len(resp) > max_response_chars:
            resp = resp[:max_response_chars]
        bundles.append(_build_bundle(
            loaded.tokenizer, template,
            r["s_instruction"], r["u_instruction"], r["condition"],
            resp, supports_thinking, system_role_supported,
        ))

    with timer(f"[{model_key}] rescore: extract+ppl over {len(bundles)} bundles"):
        acts, ppl = extract_multi_position_with_ppl_batched(
            loaded, bundles,
            position_names=positions,
            batch_size=batch_size, max_length=max_length, cache_dir=None,
        )
    # acts shape: (n_rows, n_pos, n_layers, d)

    n_rows, n_pos_act, n_layers, _ = acts.shape
    if n_pos_act != len(positions):
        raise RuntimeError(f"position-axis mismatch: got {n_pos_act}, expected {len(positions)}")

    scores_by_pos: dict[str, np.ndarray] = {}
    for pi, pos_name in enumerate(positions):
        scores = np.full((n_rows, n_layers), np.nan, dtype=np.float32)
        for layer in range(n_layers):
            d_vec = dirs_by_position[pos_name].get(layer)
            if d_vec is None:
                continue
            v = acts[:, pi, layer, :]
            norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
            v_n = v / norms
            scores[:, layer] = v_n @ d_vec
        scores_by_pos[pos_name] = scores

    if free_after:
        free_model(loaded)

    judge_lookup = _read_judge_csv(judge_csv_path) if judge_csv_path else {}

    fields = [
        "pair_idx", "condition", "conflict_axis",
        "s_instruction", "u_instruction", "response_preview",
        "best_layer", "best_position",
        "ppl_response",
    ]
    for p in positions:
        fields.append(f"probe_score_{p}_best_layer")
        fields.append(f"probe_scores_{p}_by_layer_json")
    if judge_lookup:
        fields.extend(["judge_verdict", "system_followed", "judge_reason"])

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows):
            row_out = {
                "pair_idx": r.get("pair_idx", ""),
                "condition": r.get("condition", ""),
                "conflict_axis": r.get("conflict_axis", ""),
                "s_instruction": r.get("s_instruction", ""),
                "u_instruction": r.get("u_instruction", ""),
                "response_preview": (r.get("response", "") or "")[:400],
                "best_layer": best_layer,
                "best_position": best_position or "",
                "ppl_response": "" if not np.isfinite(ppl[i]) else float(ppl[i]),
            }
            for p in positions:
                arr = scores_by_pos[p][i]
                v = arr[best_layer] if best_layer < n_layers else float("nan")
                row_out[f"probe_score_{p}_best_layer"] = "" if not np.isfinite(v) else float(v)
                psbl = {int(L): float(s) for L, s in enumerate(arr) if np.isfinite(s)}
                row_out[f"probe_scores_{p}_by_layer_json"] = json.dumps(psbl)
            if judge_lookup:
                jr = judge_lookup.get((str(r.get("pair_idx", "")), str(r.get("condition", ""))), {})
                row_out["judge_verdict"] = jr.get("judge_verdict", "")
                row_out["system_followed"] = jr.get("system_followed", "")
                row_out["judge_reason"] = jr.get("judge_reason", "")
            w.writerow(row_out)
    logger.info(f"Wrote {len(rows)} rescored rows to {out_csv_path}")

    # ----- Summary: per-position best-layer correlation with judge verdicts -----
    summary: dict = {
        "n_rows": len(rows),
        "best_layer": best_layer,
        "best_position": best_position,
        "positions": list(positions),
    }
    if judge_lookup:
        from scipy.stats import pearsonr  # lazy
        by_pos = {}
        for p in positions:
            xs, ys = [], []
            for i, r in enumerate(rows):
                jr = judge_lookup.get((str(r.get("pair_idx", "")), str(r.get("condition", ""))))
                if jr is None:
                    continue
                followed = str(jr.get("system_followed", "")).strip().lower()
                if followed not in ("true", "false"):
                    continue
                v = scores_by_pos[p][i, best_layer] if best_layer < n_layers else float("nan")
                if not np.isfinite(v):
                    continue
                xs.append(float(v))
                ys.append(1 if followed == "true" else 0)
            entry = {"n": len(xs)}
            if len(xs) >= 3 and len(set(ys)) > 1:
                r_val, p_val = pearsonr(xs, ys)
                entry["r"] = float(r_val)
                entry["p"] = float(p_val)
            by_pos[p] = entry
        summary["correlations_at_best_layer"] = by_pos
        for p, e in by_pos.items():
            r_str = f"{e['r']:+.3f}" if "r" in e else "n/a"
            p_str = f"{e['p']:.2g}" if "p" in e else "n/a"
            logger.info(f"  {p:>15s}  n={e['n']:4d}  r={r_str}  p={p_str}")
    return summary
