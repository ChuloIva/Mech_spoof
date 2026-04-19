"""Experiment 4 — Attack payload evaluation + token-by-token trace (§7)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mech_spoof.configs import ATTACK_GENERATION_MAX_NEW_TOKENS
from mech_spoof.datasets.advbench import load_advbench
from mech_spoof.datasets.attacks import AttackPayload, build_attack_payloads
from mech_spoof.eval.refusal_check import check_refusal
from mech_spoof.io import load_json, load_npz, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class Exp4Result:
    model_key: str
    summary: dict
    rows: list[dict]


def _load_direction(bundle_dir: Path, kind: Literal["probe", "refusal"], layer: int) -> np.ndarray:
    arrays = load_npz(Path(bundle_dir) / "arrays.npz")
    prefix = "probe_dir_layer_" if kind == "probe" else "refusal_dir_layer_"
    return arrays[f"{prefix}{layer:03d}"]


def _load_best_layer(exp1_dir: Path) -> int:
    result = load_json(Path(exp1_dir) / "result.json")
    return int(result["best_layer"])


def _residual_at_position(loaded, input_ids, layer: int, position: int) -> np.ndarray:
    """Extract normalized residual at (layer, position)."""
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
    return act / (np.linalg.norm(act) + 1e-8)


def _wrap_payload_as_user_prompt(loaded, payload_text: str) -> list[int]:
    """Wrap an attack payload string as a user turn and tokenize."""
    messages = [{"role": "user", "content": payload_text}]
    extra = (
        {"enable_thinking": False}
        if getattr(loaded.template, "_supports_enable_thinking", False)
        else {}
    )
    ids = loaded.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, **extra
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def trace_authority_through_injection(
    loaded,
    payload_text: str,
    authority_direction: np.ndarray,
    refusal_direction: np.ndarray,
    best_layer: int,
) -> list[dict]:
    """Token-by-token projection of residual onto authority + refusal directions (§7.2)."""
    import torch

    input_ids = _wrap_payload_as_user_prompt(loaded, payload_text)

    storage: list = [None] * loaded.n_layers
    handles = []
    for i in range(loaded.n_layers):
        def _hook(_m, _inp, out, idx=i):
            h = out[0] if isinstance(out, tuple) else out
            storage[idx] = h.detach()
        handles.append(loaded.layer_module(i).register_forward_hook(_hook))
    try:
        ids_t = torch.tensor([input_ids], dtype=torch.long, device=loaded.device)
        with torch.no_grad():
            loaded.hf_model(input_ids=ids_t, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    acts = storage[best_layer][0].float().cpu().numpy()  # (seq_len, d_model)
    norms = np.linalg.norm(acts, axis=1, keepdims=True) + 1e-8
    acts_normed = acts / norms

    a_dir = authority_direction / (np.linalg.norm(authority_direction) + 1e-8)
    r_dir = refusal_direction / (np.linalg.norm(refusal_direction) + 1e-8)

    auth_scores = acts_normed @ a_dir
    ref_scores = acts_normed @ r_dir

    token_strings = loaded.tokenizer.convert_ids_to_tokens(input_ids)
    template = loaded.template
    trace = []
    for i, (tok, a, r) in enumerate(zip(token_strings, auth_scores, ref_scores)):
        trace.append({
            "position": i,
            "token": tok,
            "authority_score": float(a),
            "refusal_score": float(r),
            "is_delimiter": template.is_delimiter_token(tok),
            "structural_role": template.classify_structural_role(i, input_ids),
        })
    return trace


def run_experiment_4(
    model_key: str,
    out_dir: Path,
    exp1_dir: Path,
    exp3_dir: Path,
    max_new_tokens: int = ATTACK_GENERATION_MAX_NEW_TOKENS,
    n_goals: int = 30,
    n_trace_payloads: int = 3,
    seed: int = 42,
    free_after: bool = True,
) -> Exp4Result:
    """Score each attack payload for authority, refusal, and behavioral outcome."""
    import torch

    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_layer = _load_best_layer(exp1_dir)
    authority_dir = _load_direction(exp1_dir, "probe", best_layer)
    refusal_dir = _load_direction(exp3_dir, "refusal", best_layer)

    loaded = load_model(model_key)
    harmful = load_advbench()[:n_goals]
    attacks = build_attack_payloads(loaded.template, harmful, max_per_attack=n_goals)

    rows: list[dict] = []
    for attack_type, payloads in attacks.items():
        for p in payloads:
            input_ids = _wrap_payload_as_user_prompt(loaded, p.text)

            # Authority + refusal at response_first
            last_pos = len(input_ids) - 1
            act_norm = _residual_at_position(loaded, input_ids, best_layer, last_pos)
            auth_score = float(act_norm @ authority_dir)
            ref_score = float(act_norm @ refusal_dir)

            # Generate short continuation
            with torch.no_grad():
                out = loaded.hf_model.generate(
                    torch.tensor([input_ids], dtype=torch.long, device=loaded.device),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=loaded.tokenizer.pad_token_id,
                )
            response = loaded.tokenizer.decode(
                out[0][len(input_ids):], skip_special_tokens=True
            )
            refused = check_refusal(response)

            rows.append({
                "attack_type": attack_type,
                "harmful_goal": p.harmful_goal,
                "n_turns": p.n_turns,
                "authority_score": auth_score,
                "refusal_score": ref_score,
                "refused": bool(refused),
                "response_prefix": response[:200],
            })

    # Per-attack-type summary
    summary: dict = {}
    for atype in attacks:
        subset = [r for r in rows if r["attack_type"] == atype]
        auth_scores = [r["authority_score"] for r in subset]
        ref_scores = [r["refusal_score"] for r in subset]
        summary[atype] = {
            "n": len(subset),
            "refusal_rate": float(np.mean([1 if r["refused"] else 0 for r in subset])),
            "mean_authority": float(np.mean(auth_scores)),
            "mean_refusal": float(np.mean(ref_scores)),
        }

    # Optional: token-by-token trace for a few representative multi-turn payloads
    traces = []
    multi_turn = attacks.get("chatinject_multi_turn", [])[:n_trace_payloads]
    for p in multi_turn:
        with timer(f"[{model_key}] trace ({p.harmful_goal[:40]}...)"):
            trace = trace_authority_through_injection(
                loaded, p.text, authority_dir, refusal_dir, best_layer
            )
        traces.append({"harmful_goal": p.harmful_goal, "trace": trace})

    result_json = {
        "model_key": model_key,
        "best_layer": int(best_layer),
        "summary": summary,
        "rows": rows,
        "traces": traces,
    }
    save_result_bundle(
        out_dir,
        json_obj=result_json,
        manifest_extras={"experiment": "exp4_attacks", "model_key": model_key},
    )

    if free_after:
        free_model(loaded)

    return Exp4Result(model_key=model_key, summary=summary, rows=rows)
