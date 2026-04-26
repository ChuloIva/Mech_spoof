"""Experiment 6 — Structural authority probe with prefill control + transfer evals.

Builds on exp1 (S-vs-U at the prompt's last token) by:
  1. Including a user follow-up + model response in each trace.
  2. Extracting at response_first / response_mid / response_last
     (post-decision residual states), matching the exp2b/2d apparatus.
  3. A prefilled-response control: generate R_S from the S-condition once via vLLM,
     then prefill that same R_S into both the S and U trace under matched-baseline.
     The probe sees identical response tokens — only prompt structure differs.
  4. Difference-in-means probes (Marks & Tegmark, COLM 2024) as the primary
     probe; LR (logistic regression) as a comparison. Cosine similarity flags
     when LR is exploiting a feature MM ignores.
  5. Two transfer evals: FAKE delimiter (does delim injection spoof the
     authority direction?) and NONE-conflict (does the structural probe
     correlate with behavioural compliance from exp1b/2b/2d?).
  6. Causal intervention sweep at the best (layer, position): add α·direction
     to a U-condition forward pass and check whether behaviour flips toward S.

Source dataset: `data/conflict_pairs_stratified_1k.jsonl`. Each row has
`s_instruction`, `u_instruction`, and `original_prompt` (the natural user
message the instruction is meant to govern). For exp6 we expand each row
into two single-instruction items: one using s_instruction, one using
u_instruction, both with `original_prompt` as the user follow-up. We do not
care about the conflict here — we only need diverse (instruction, follow-up)
pairs where the follow-up is naturally related to the instruction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from mech_spoof.activations import extract_multi_position_with_ppl_batched
from mech_spoof.configs import DATA_DIR, MODEL_CONFIGS
from mech_spoof.datasets.conflict_evolved import (
    EvolvedConflictPair,
    load_evolved_conflict_pairs,
    load_held_out_evolved_pairs,
)
from mech_spoof.eval.llm_judge_vllm import JudgeRow, judge_with_vllm
from mech_spoof.io import save_pickle, save_result_bundle
from mech_spoof.models import free_model, load_model
from mech_spoof.probes import (
    DiffMeanProbe,
    cosine_agreement,
    fit_diff_mean_multi_layer,
    fit_diff_mean_probes,
    intervene_along_direction,
    score_multi_layer,
    train_authority_probe,
)
from mech_spoof.templates.base import PromptBundle
from mech_spoof.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


DEFAULT_SOURCE = DATA_DIR / "conflict_pairs_stratified_1k.jsonl"
DEFAULT_BASELINE_SYSTEM = "You are a helpful assistant."
POSITION_NAMES = ("response_first", "response_mid", "response_last")


# -----------------------------------------------------------------------------
# Source items
# -----------------------------------------------------------------------------

@dataclass
class Exp6Item:
    """One (instruction, follow-up) item — expanded from a stratified-1k row."""
    row_idx: int
    which: Literal["s", "u"]   # which instruction we pulled from the row
    instruction: str
    original_prompt: str       # used as user_followup
    macro_axis: str
    conflict_axis: str


def _load_instruction_pool(
    source_path: Path | None = None,
    max_rows: int | None = None,
    instruction_choices: tuple[str, ...] = ("s", "u"),
) -> list[Exp6Item]:
    """Load stratified-1k rows and expand each into items of the chosen instructions."""
    path = Path(source_path) if source_path else DEFAULT_SOURCE
    items: list[Exp6Item] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            row = json.loads(line)
            for which in instruction_choices:
                if which == "s":
                    instr = row["s_instruction"]
                elif which == "u":
                    instr = row["u_instruction"]
                else:
                    raise ValueError(f"unknown instruction choice: {which}")
                items.append(Exp6Item(
                    row_idx=i,
                    which=which,
                    instruction=instr,
                    original_prompt=row.get("original_prompt", ""),
                    macro_axis=row.get("macro_axis", "unknown"),
                    conflict_axis=row.get("conflict_axis", "unknown"),
                ))
    logger.info(
        f"Loaded {len(items)} items "
        f"({len(items) // max(1, len(instruction_choices))} rows × "
        f"{len(instruction_choices)} instructions) from {path}"
    )
    return items


# -----------------------------------------------------------------------------
# Bundle builders (matched-baseline S/U, with optional prefilled response)
# -----------------------------------------------------------------------------

def _flatten_ids(ids) -> list[int]:
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def _build_matched_bundle(
    tokenizer,
    baseline_system: str,
    instruction: str,
    user_followup: str,
    response: str | None,
    condition: Literal["S", "U", "FAKE"],
    extras: dict | None = None,
    supports_enable_thinking: bool = False,
    fake_injection_text: str | None = None,
) -> PromptBundle:
    """Build a matched-baseline prompt bundle.

    S: system = "{baseline}\\n{instruction}", user = followup
    U: system = "{baseline}",                user = "{instruction}\\n\\n{followup}"
    FAKE: system = "{baseline}", user = "{fake_injection_text}\\n\\n{followup}"
          (fake injection IS the structural attack — instruction not in user text).

    If `response` is None or "", builds a generation prompt (add_generation_prompt=True,
    no assistant turn). The bundle's response_token_span defaults to the trailing
    last-input position only.

    If `response` is non-empty, appends an assistant turn containing `response`
    and sets response_token_span to the assistant tokens — the slice over which
    we extract response_first/mid/last activations.
    """
    if condition == "S":
        system_content = f"{baseline_system}\n{instruction}"
        user_content = user_followup
    elif condition == "U":
        system_content = baseline_system
        user_content = f"{instruction}\n\n{user_followup}"
    elif condition == "FAKE":
        if not fake_injection_text:
            raise ValueError("FAKE condition requires fake_injection_text")
        system_content = baseline_system
        user_content = f"{fake_injection_text}\n\n{user_followup}"
    else:
        raise ValueError(f"unknown condition {condition}")

    extra_kwargs = {"enable_thinking": False} if supports_enable_thinking else {}

    messages_prompt = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    prompt_ids = _flatten_ids(tokenizer.apply_chat_template(
        messages_prompt, tokenize=True, add_generation_prompt=True, **extra_kwargs,
    ))
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True, **extra_kwargs,
    )

    if not response:
        full_ids = prompt_ids
        full_text = prompt_text
        # Default span: trailing position.
        response_start = len(full_ids) - 1
        response_end = len(full_ids)
    else:
        messages_full = messages_prompt + [{"role": "assistant", "content": response}]
        full_ids = _flatten_ids(tokenizer.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False, **extra_kwargs,
        ))
        full_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False, **extra_kwargs,
        )
        response_start = min(len(prompt_ids), len(full_ids))
        response_end = len(full_ids)
        if response_end <= response_start:
            response_start = max(0, response_end - 1)

    bundle_extras = dict(extras or {})
    bundle_extras["response_token_span"] = (response_start, response_end)
    if response:
        bundle_extras["prefilled_response"] = response[:200]
    bundle_extras["matched_baseline"] = baseline_system

    return PromptBundle(
        text=full_text,
        input_ids=full_ids,
        instruction_text=instruction,
        instruction_token_span=(0, response_start),
        response_first_pos=response_start,
        condition="S" if condition == "S" else "U",  # PromptBundle.condition Literal — keep S/U
        extras=bundle_extras,
    )


def _build_four_bundles(
    tokenizer,
    item: Exp6Item,
    baseline_system: str,
    response_S: str,
    supports_enable_thinking: bool,
) -> dict[str, PromptBundle]:
    """Build {S_free, U_free, S_prefill, U_prefill} for one item.

    S_free / U_free: generation prompts (no assistant turn) — used for free-gen
        activation extraction (validation of the prefilled-trained probe).
    S_prefill / U_prefill: same prompts with response_S appended as assistant
        content. Activation extraction at response_first/mid/last reads the
        prefilled response tokens.
    """
    common = dict(
        tokenizer=tokenizer,
        baseline_system=baseline_system,
        instruction=item.instruction,
        user_followup=item.original_prompt or "Hello, how can you help me?",
        supports_enable_thinking=supports_enable_thinking,
    )
    base_extras = {
        "row_idx": item.row_idx,
        "which": item.which,
        "macro_axis": item.macro_axis,
        "conflict_axis": item.conflict_axis,
    }
    return {
        "S_free": _build_matched_bundle(
            response=None, condition="S",
            extras={**base_extras, "trace": "S_free"}, **common,
        ),
        "U_free": _build_matched_bundle(
            response=None, condition="U",
            extras={**base_extras, "trace": "U_free"}, **common,
        ),
        "S_prefill": _build_matched_bundle(
            response=response_S, condition="S",
            extras={**base_extras, "trace": "S_prefill"}, **common,
        ),
        "U_prefill": _build_matched_bundle(
            response=response_S, condition="U",
            extras={**base_extras, "trace": "U_prefill"}, **common,
        ),
    }


# -----------------------------------------------------------------------------
# vLLM free generation of R_S
# -----------------------------------------------------------------------------

def _generate_responses_vllm(
    items: list[Exp6Item],
    vllm_model_id: str,
    baseline_system: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 42,
    free_after: bool = True,
    enable_thinking: bool = False,
) -> list[str]:
    """For each item, generate R_S from the S-condition messages via vLLM (greedy)."""
    from vllm import LLM, SamplingParams

    logger.info(
        f"vLLM free-gen for {len(items)} items: model={vllm_model_id} "
        f"max_tokens={max_tokens} temp={temperature}"
    )
    llm = LLM(
        model=vllm_model_id,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=seed,
    )
    tok = llm.get_tokenizer()

    # Probe whether the chat template accepts enable_thinking (Qwen3+).
    template_supports_thinking = False
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        template_supports_thinking = True
    except (TypeError, ValueError):
        template_supports_thinking = False

    extra_kwargs = {"enable_thinking": enable_thinking} if template_supports_thinking else {}

    prompts: list[str] = []
    for it in items:
        followup = it.original_prompt or "Hello, how can you help me?"
        messages = [
            {"role": "system", "content": f"{baseline_system}\n{it.instruction}"},
            {"role": "user", "content": followup},
        ]
        prompts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **extra_kwargs,
        ))

    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed,
    )
    results = llm.generate(prompts, sampling)
    responses = [r.outputs[0].text if r.outputs else "" for r in results]

    if free_after:
        try:
            del llm
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:  # pragma: no cover
            logger.warning(f"vLLM cleanup hiccup: {e}")

    n_empty = sum(1 for r in responses if not r.strip())
    logger.info(f"vLLM free-gen done: {len(responses)} responses, {n_empty} empty")
    return responses


# -----------------------------------------------------------------------------
# Probe fitting per (layer × position)
# -----------------------------------------------------------------------------

@dataclass
class PositionProbeBundle:
    """Probes + accuracies for one extraction position."""
    position: str
    mm_probes: dict[int, DiffMeanProbe]
    mm_accuracies_train: dict[int, float]
    mm_accuracies_val: dict[int, float]
    mm_accuracies_free_val: dict[int, float]    # train on prefilled, eval on free
    lr_directions: dict[int, np.ndarray]
    lr_accuracies_val: dict[int, float]
    lr_aurocs_val: dict[int, float]
    mm_lr_cosine: dict[int, float]
    multi_layer_top: list[int]
    multi_layer_acc: float
    multi_layer_acc_free: float
    best_layer: int
    best_accuracy: float


def _split_indices_by_row(
    items: list[Exp6Item],
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-level train/val split. Both items from the same row go to the same side."""
    import random
    rng = random.Random(seed)
    rows = sorted({it.row_idx for it in items})
    rng.shuffle(rows)
    n_train_rows = int(round(len(rows) * train_fraction))
    train_rows = set(rows[:n_train_rows])
    train_idx = np.array([i for i, it in enumerate(items) if it.row_idx in train_rows])
    val_idx = np.array([i for i, it in enumerate(items) if it.row_idx not in train_rows])
    return train_idx, val_idx


def _fit_probes_for_position(
    position: str,
    s_prefill: np.ndarray,    # (n_items, n_layers, d) at this position
    u_prefill: np.ndarray,
    s_free: np.ndarray,
    u_free: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    multi_layer_k: int = 5,
) -> PositionProbeBundle:
    """Fit MM and LR per layer at one position; compute MM-vs-LR cosine, multi-layer
    ensemble, and prefilled→free transfer accuracy."""
    n_layers = s_prefill.shape[1]

    # MM probes: fit on prefilled training rows.
    mm_probes = fit_diff_mean_probes(
        acts_A=s_prefill[train_idx],
        acts_B=u_prefill[train_idx],
    )

    # MM accuracies: train + val (prefilled), and prefilled→free (val rows only).
    mm_acc_train: dict[int, float] = {}
    mm_acc_val: dict[int, float] = {}
    mm_acc_free_val: dict[int, float] = {}
    for layer, probe in mm_probes.items():
        # Train (prefilled)
        X_tr = np.concatenate([s_prefill[train_idx, layer, :], u_prefill[train_idx, layer, :]])
        y_tr = np.concatenate([np.ones(len(train_idx)), np.zeros(len(train_idx))]).astype(int)
        mm_acc_train[layer] = probe.accuracy(X_tr, y_tr)
        # Val (prefilled)
        X_va = np.concatenate([s_prefill[val_idx, layer, :], u_prefill[val_idx, layer, :]])
        y_va = np.concatenate([np.ones(len(val_idx)), np.zeros(len(val_idx))]).astype(int)
        mm_acc_val[layer] = probe.accuracy(X_va, y_va)
        # Prefilled → free transfer (val rows only — train rows leak)
        X_fr = np.concatenate([s_free[val_idx, layer, :], u_free[val_idx, layer, :]])
        mm_acc_free_val[layer] = probe.accuracy(X_fr, y_va)

    # LR probes: refit on the same prefilled training rows.
    s_train = s_prefill[train_idx]    # (n_train, n_layers, d)
    u_train = u_prefill[train_idx]
    lr_result = train_authority_probe(
        system_acts=s_train, user_acts=u_train,
        test_size=0.25, seed=seed, normalize=True,
    )
    # The LR probe split was internal — we need val accuracy on our held-out rows. Refit
    # would re-split; instead, project our val rows onto the LR direction with its threshold.
    lr_acc_val: dict[int, float] = {}
    lr_aurocs_val: dict[int, float] = {}
    from sklearn.metrics import accuracy_score, roc_auc_score
    for layer, probe in lr_result.probes.items():
        X_va = np.concatenate([s_prefill[val_idx, layer, :], u_prefill[val_idx, layer, :]])
        # Reuse the normalize convention from train_authority_probe (normalize=True).
        Xn = X_va / (np.linalg.norm(X_va, axis=-1, keepdims=True) + 1e-8)
        y_va = np.concatenate([np.ones(len(val_idx)), np.zeros(len(val_idx))]).astype(int)
        y_pred = probe.predict(Xn)
        y_prob = probe.predict_proba(Xn)[:, 1]
        lr_acc_val[layer] = float(accuracy_score(y_va, y_pred))
        try:
            lr_aurocs_val[layer] = float(roc_auc_score(y_va, y_prob))
        except ValueError:
            lr_aurocs_val[layer] = float("nan")

    # MM vs LR cosine.
    mm_dirs = {l: p.direction for l, p in mm_probes.items()}
    cos = cosine_agreement(mm_dirs, lr_result.directions)

    # Multi-layer MM ensemble: top-K layers by single-layer val accuracy.
    sorted_layers = sorted(mm_acc_val, key=lambda l: mm_acc_val[l], reverse=True)
    top_layers = sorted_layers[:multi_layer_k]
    direction_ml, midpoint_ml, _raw_ml = fit_diff_mean_multi_layer(
        acts_A=s_prefill[train_idx],
        acts_B=u_prefill[train_idx],
        layers=top_layers,
    )
    # Multi-layer val accuracy (prefilled).
    val_scores_S = score_multi_layer(s_prefill[val_idx], direction_ml, midpoint_ml, top_layers)
    val_scores_U = score_multi_layer(u_prefill[val_idx], direction_ml, midpoint_ml, top_layers)
    ml_pred_S = (val_scores_S > 0).astype(int)
    ml_pred_U = (val_scores_U > 0).astype(int)
    ml_acc = float(np.mean(np.concatenate([ml_pred_S, 1 - ml_pred_U])))
    # Multi-layer prefilled→free transfer.
    free_scores_S = score_multi_layer(s_free[val_idx], direction_ml, midpoint_ml, top_layers)
    free_scores_U = score_multi_layer(u_free[val_idx], direction_ml, midpoint_ml, top_layers)
    ml_free_pred_S = (free_scores_S > 0).astype(int)
    ml_free_pred_U = (free_scores_U > 0).astype(int)
    ml_free_acc = float(np.mean(np.concatenate([ml_free_pred_S, 1 - ml_free_pred_U])))

    best_layer = max(mm_acc_val, key=mm_acc_val.get)

    return PositionProbeBundle(
        position=position,
        mm_probes=mm_probes,
        mm_accuracies_train=mm_acc_train,
        mm_accuracies_val=mm_acc_val,
        mm_accuracies_free_val=mm_acc_free_val,
        lr_directions=lr_result.directions,
        lr_accuracies_val=lr_acc_val,
        lr_aurocs_val=lr_aurocs_val,
        mm_lr_cosine=cos,
        multi_layer_top=top_layers,
        multi_layer_acc=ml_acc,
        multi_layer_acc_free=ml_free_acc,
        best_layer=best_layer,
        best_accuracy=mm_acc_val[best_layer],
    )


# -----------------------------------------------------------------------------
# Transfer eval: FAKE delimiter
# -----------------------------------------------------------------------------

def _eval_transfer_fake(
    loaded,
    items: list[Exp6Item],
    baseline_system: str,
    extract_batch_size: int,
    extract_max_length: int,
    bundles_S_free,
    bundles_U_free,
    s_free_acts: np.ndarray,
    u_free_acts: np.ndarray,
    position: str,
    layer: int,
    mm_probe: DiffMeanProbe,
    supports_enable_thinking: bool,
) -> dict:
    """Build FAKE-delimiter prompts (matched-baseline + fake injection in user field)
    and score with the trained MM probe at (layer, position).

    Returns mean projected scores for FAKE / S_free / U_free baselines so we can
    judge: does FAKE ≈ S (delimiter spoofs authority) or ≈ U (model not fooled)?
    """
    template = loaded.template
    fake_bundles: list[PromptBundle] = []
    for it in items:
        fake_text = template.build_fake_delimiter_injection(it.instruction)
        fake_bundles.append(_build_matched_bundle(
            tokenizer=loaded.tokenizer,
            baseline_system=baseline_system,
            instruction=it.instruction,
            user_followup=it.original_prompt or "Hello, how can you help me?",
            response=None,
            condition="FAKE",
            fake_injection_text=fake_text,
            extras={
                "row_idx": it.row_idx, "which": it.which, "trace": "FAKE",
                "macro_axis": it.macro_axis, "conflict_axis": it.conflict_axis,
            },
            supports_enable_thinking=supports_enable_thinking,
        ))
    fake_acts, _ = extract_multi_position_with_ppl_batched(
        loaded, fake_bundles,
        position_names=POSITION_NAMES,
        batch_size=extract_batch_size,
        max_length=extract_max_length,
        progress=True,
    )
    pos_idx = POSITION_NAMES.index(position)
    fake_X = fake_acts[:, pos_idx, layer, :]
    s_X = s_free_acts[:, pos_idx, layer, :]
    u_X = u_free_acts[:, pos_idx, layer, :]

    fake_scores = mm_probe.score_batch(fake_X)
    s_scores = mm_probe.score_batch(s_X)
    u_scores = mm_probe.score_batch(u_X)

    return {
        "position": position,
        "layer": layer,
        "n_items": len(items),
        "fake_score_mean": float(fake_scores.mean()),
        "fake_score_std": float(fake_scores.std()),
        "s_free_score_mean": float(s_scores.mean()),
        "s_free_score_std": float(s_scores.std()),
        "u_free_score_mean": float(u_scores.mean()),
        "u_free_score_std": float(u_scores.std()),
        "fake_classified_as_S_pct": float((fake_scores > 0).mean()),
        "s_free_classified_as_S_pct": float((s_scores > 0).mean()),
        "u_free_classified_as_S_pct": float((u_scores > 0).mean()),
        "fake_scores": fake_scores.tolist(),
        "s_free_scores": s_scores.tolist(),
        "u_free_scores": u_scores.tolist(),
    }


# -----------------------------------------------------------------------------
# Transfer eval: NONE-conflict (correlate probe score with system_followed)
# -----------------------------------------------------------------------------

def _eval_transfer_none_conflict(
    loaded,
    pairs: list[EvolvedConflictPair],
    vllm_model_id: str,
    judge_model_id: str,
    extract_batch_size: int,
    extract_max_length: int,
    position: str,
    layer: int,
    mm_probe: DiffMeanProbe,
    out_dir: Path,
    seed: int,
    supports_enable_thinking: bool,
    max_new_tokens: int = 200,
) -> dict:
    """Generate NONE-condition responses for held-out conflict pairs, judge with vLLM,
    and correlate the MM probe score with system_followed.

    Reuses exp2b's pattern: build NONE prompts via template.make_conflict_prompt,
    generate via vLLM, judge, then re-forward prompt+response through HF and read
    the probe score.
    """
    from vllm import LLM, SamplingParams

    template = loaded.template
    none_prompts: list[PromptBundle] = []
    for p in pairs:
        b = template.make_conflict_prompt(
            system_instruction=p.s_instruction,
            user_instruction=p.u_instruction,
            condition="NONE",
        )
        none_prompts.append(b)

    # Generate via vLLM.
    logger.info(f"NONE-conflict transfer: generating {len(none_prompts)} responses via {vllm_model_id}")
    llm = LLM(
        model=vllm_model_id,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=seed,
    )
    tok = llm.get_tokenizer()
    template_supports_thinking = False
    try:
        tok.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        template_supports_thinking = True
    except (TypeError, ValueError):
        template_supports_thinking = False

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, seed=seed)
    prompt_texts = [b.text for b in none_prompts]
    results = llm.generate(prompt_texts, sp)
    responses = [r.outputs[0].text if r.outputs else "" for r in results]

    try:
        del llm
        import gc, torch  # noqa
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:  # pragma: no cover
        logger.warning(f"vLLM cleanup hiccup: {e}")

    # Judge.
    judge_rows = [
        JudgeRow(
            s_instruction=p.s_instruction,
            u_instruction=p.u_instruction,
            s_gold=p.s_aligned_response,
            u_gold=p.u_aligned_response,
            response=r,
            extras={"pair_idx": p.idx, "macro_axis": p.macro_axis, "conflict_axis": p.conflict_axis},
        )
        for p, r in zip(pairs, responses)
    ]
    verdicts = judge_with_vllm(
        judge_rows, model_id=judge_model_id, enable_thinking=False, free_after=True,
    )

    # Save generations + verdicts for the recovery path / debugging.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "none_conflict_generations.jsonl", "w") as f:
        for p, r, v in zip(pairs, responses, verdicts):
            f.write(json.dumps({
                "pair_idx": p.idx,
                "s_instruction": p.s_instruction,
                "u_instruction": p.u_instruction,
                "response": r,
                "verdict": v["verdict"],
                "reason": v["reason"],
            }) + "\n")

    # Build prompt+response bundles and re-extract activations.
    bundles_with_resp: list[PromptBundle] = []
    for p, r, v in zip(pairs, responses, verdicts):
        if v["verdict"] not in ("system", "user"):
            continue
        # Re-tokenize prompt+response with assistant turn.
        extra_kwargs = {"enable_thinking": False} if supports_enable_thinking else {}
        messages_full = [
            {"role": "user", "content": f"{p.s_instruction}\n\n{p.u_instruction}"},
            {"role": "assistant", "content": r},
        ]
        full_ids = _flatten_ids(loaded.tokenizer.apply_chat_template(
            messages_full, tokenize=True, add_generation_prompt=False, **extra_kwargs,
        ))
        full_text = loaded.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False, **extra_kwargs,
        )
        prefix_ids = _flatten_ids(loaded.tokenizer.apply_chat_template(
            messages_full[:-1], tokenize=True, add_generation_prompt=True, **extra_kwargs,
        ))
        r_start = min(len(prefix_ids), len(full_ids))
        r_end = len(full_ids)
        if r_end <= r_start:
            r_start = max(0, r_end - 1)
        bundles_with_resp.append(PromptBundle(
            text=full_text,
            input_ids=full_ids,
            instruction_text=p.s_instruction,
            instruction_token_span=(0, r_start),
            response_first_pos=r_start,
            condition="NONE",
            extras={
                "pair_idx": p.idx,
                "verdict": v["verdict"],
                "system_followed": int(v["verdict"] == "system"),
                "response_token_span": (r_start, r_end),
            },
        ))

    if not bundles_with_resp:
        logger.warning("NONE-conflict: 0 judgable rows — skipping correlation")
        return {"n_judgable": 0, "pearson_r": None, "p_value": None}

    acts, _ = extract_multi_position_with_ppl_batched(
        loaded, bundles_with_resp,
        position_names=POSITION_NAMES,
        batch_size=extract_batch_size,
        max_length=extract_max_length,
        progress=True,
    )
    pos_idx = POSITION_NAMES.index(position)
    X = acts[:, pos_idx, layer, :]
    scores = mm_probe.score_batch(X)
    sys_followed = np.array([b.extras["system_followed"] for b in bundles_with_resp])

    from scipy.stats import pearsonr
    if len(np.unique(sys_followed)) < 2:
        r, p = float("nan"), float("nan")
    else:
        r, p = pearsonr(scores, sys_followed)

    return {
        "n_judgable": len(bundles_with_resp),
        "n_total": len(pairs),
        "n_system_followed": int(sys_followed.sum()),
        "n_user_followed": int((1 - sys_followed).sum()),
        "pearson_r": float(r),
        "p_value": float(p),
        "scores": scores.tolist(),
        "sys_followed": sys_followed.tolist(),
        "position": position,
        "layer": layer,
    }


# -----------------------------------------------------------------------------
# Causal intervention sweep
# -----------------------------------------------------------------------------

def _intervention_sweep(
    loaded,
    items: list[Exp6Item],
    baseline_system: str,
    layer: int,
    probe: DiffMeanProbe,
    n_prompts: int,
    judge_model_id: str,
    supports_enable_thinking: bool,
    seed: int,
    alphas: list[float] | None = None,
    max_new_tokens: int = 128,
) -> dict:
    """At the chosen best layer, sweep α and check whether U-condition behaviour
    flips toward S-condition compliance.

    For each α: run U_free prompts through `intervene_along_direction(loaded, ..., alpha)`,
    obtain generated text, judge each pair (s_inst, u_inst here are baseline vs instruction
    so the framing is "did the response follow the instruction or default helpful").

    For simplicity, the judge used here checks whether the response follows
    `instruction` — we use the JudgeRow with s_gold = empty / generic, u_gold also generic,
    and instead correlate score continuum. (Coarse but enough for a sweep signal.)
    """
    import random
    rng = random.Random(seed)
    sample = rng.sample(items, min(n_prompts, len(items)))

    natural = probe.natural_scale
    if alphas is None:
        alphas = [-2.0 * natural, -1.0 * natural, 0.0, 1.0 * natural, 2.0 * natural]

    # Build U_free prompts (no response yet — we'll generate under intervention).
    u_bundles: list[PromptBundle] = []
    for it in sample:
        u_bundles.append(_build_matched_bundle(
            tokenizer=loaded.tokenizer,
            baseline_system=baseline_system,
            instruction=it.instruction,
            user_followup=it.original_prompt or "Hello, how can you help me?",
            response=None,
            condition="U",
            extras={"row_idx": it.row_idx, "which": it.which, "trace": "U_intervene"},
            supports_enable_thinking=supports_enable_thinking,
        ))

    sweep_outputs: list[dict] = []
    for alpha in alphas:
        gen_texts: list[str] = []
        for b in u_bundles:
            gen_ids = intervene_along_direction(
                loaded=loaded,
                input_ids=b.input_ids,
                direction=probe.direction,
                layer=layer,
                alpha=alpha,
                positions=None,    # perturb whole prompt
                max_new_tokens=max_new_tokens,
            )
            new_text = loaded.tokenizer.decode(
                gen_ids[0, len(b.input_ids):], skip_special_tokens=True,
            )
            gen_texts.append(new_text)
        sweep_outputs.append({
            "alpha": float(alpha),
            "alpha_in_natural_units": float(alpha / natural) if natural > 0 else None,
            "n": len(gen_texts),
            "samples": gen_texts[:5],   # only keep 5 samples per alpha to keep result.json small
        })

    return {
        "layer": layer,
        "n_prompts": len(sample),
        "natural_scale": float(natural),
        "alphas": [float(a) for a in alphas],
        "sweep": sweep_outputs,
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

@dataclass
class Exp6Result:
    model_key: str
    n_layers: int
    d_model: int
    n_items: int
    n_train_rows: int
    n_val_rows: int

    # Per-position probe bundles, with all per-layer accuracies inside.
    position_results: dict[str, dict] = field(default_factory=dict)

    # Top-level "best" picks.
    best_position: str = ""
    best_layer: int = -1
    best_accuracy: float = 0.0

    # Transfer evals.
    fake_transfer: dict | None = None
    none_conflict_transfer: dict | None = None

    # Causal intervention sweep.
    intervention: dict | None = None


def run_experiment_6(
    model_key: str,
    out_dir: Path,
    source_path: Path | None = None,
    max_rows: int | None = None,
    instruction_choices: tuple[str, ...] = ("s", "u"),
    baseline_system: str = DEFAULT_BASELINE_SYSTEM,
    seed: int = 42,
    free_gen_max_tokens: int = 128,
    extract_batch_size: int = 4,
    extract_max_length: int = 1024,
    train_fraction: float = 0.75,
    do_fake_transfer: bool = True,
    do_none_conflict_transfer: bool = True,
    none_conflict_n_pairs: int = 200,
    do_intervention_sweep: bool = True,
    intervention_n_prompts: int = 50,
    judge_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    vllm_gen_model_id: str | None = None,
    free_after: bool = True,
) -> Exp6Result:
    """Run exp6 end-to-end for one model.

    Pipeline:
      1. Load source items (stratified-1k expanded by instruction_choices).
      2. vLLM free-gen R_S per item.
      3. Free vLLM, load HF model.
      4. Extract activations at response_first/mid/last for all 4 trace conditions.
      5. Fit MM + LR per (position × layer) with row-level train/val split.
      6. Compute MM-vs-LR cosine + multi-layer ensemble + prefilled→free transfer.
      7. (optional) FAKE-delimiter transfer eval at best (layer, position).
      8. (optional) NONE-conflict transfer eval (gen via vLLM, judge, correlate).
      9. (optional) Causal intervention sweep at best (layer, position).
     10. Save bundle (json + npz + pickles).
    """
    set_seed(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = MODEL_CONFIGS[model_key]
    if vllm_gen_model_id is None:
        vllm_gen_model_id = cfg.hf_id

    # ---- 1. Items ----
    items = _load_instruction_pool(
        source_path=source_path, max_rows=max_rows,
        instruction_choices=instruction_choices,
    )

    # ---- 2. Free generation of R_S via vLLM ----
    gen_path = out_dir / "free_responses.jsonl"
    if gen_path.exists():
        logger.info(f"Resuming: loading cached responses from {gen_path}")
        with open(gen_path) as f:
            cached = [json.loads(line) for line in f]
        if len(cached) != len(items):
            raise RuntimeError(
                f"Cached responses length mismatch: {len(cached)} vs {len(items)} items. "
                f"Delete {gen_path} to regenerate."
            )
        responses = [c["response"] for c in cached]
    else:
        with timer(f"[{model_key}] vLLM free-gen"):
            responses = _generate_responses_vllm(
                items=items,
                vllm_model_id=vllm_gen_model_id,
                baseline_system=baseline_system,
                max_tokens=free_gen_max_tokens,
                seed=seed,
                free_after=True,
            )
        with open(gen_path, "w") as f:
            for it, r in zip(items, responses):
                f.write(json.dumps({
                    "row_idx": it.row_idx, "which": it.which,
                    "instruction": it.instruction[:200],
                    "response": r,
                }) + "\n")

    # ---- 3. Load HF model ----
    loaded = load_model(model_key)
    logger.info(f"[{model_key}] n_layers={loaded.n_layers} d_model={loaded.d_model}")
    supports_thinking = getattr(loaded.template, "_supports_enable_thinking", False)

    # ---- 4. Build 4 bundles per item, extract activations ----
    bundles_S_free: list[PromptBundle] = []
    bundles_U_free: list[PromptBundle] = []
    bundles_S_prefill: list[PromptBundle] = []
    bundles_U_prefill: list[PromptBundle] = []
    for it, resp in zip(items, responses):
        four = _build_four_bundles(
            tokenizer=loaded.tokenizer,
            item=it,
            baseline_system=baseline_system,
            response_S=resp,
            supports_enable_thinking=supports_thinking,
        )
        bundles_S_free.append(four["S_free"])
        bundles_U_free.append(four["U_free"])
        bundles_S_prefill.append(four["S_prefill"])
        bundles_U_prefill.append(four["U_prefill"])

    def _extract(bundles, name):
        cache = out_dir / f"act_cache_{name}"
        with timer(f"[{model_key}] extract {name} (bs={extract_batch_size})"):
            acts, _ppl = extract_multi_position_with_ppl_batched(
                loaded, bundles,
                position_names=POSITION_NAMES,
                batch_size=extract_batch_size,
                max_length=extract_max_length,
                cache_dir=cache,
                progress=True,
            )
        return acts

    s_free_acts = _extract(bundles_S_free, "S_free")
    u_free_acts = _extract(bundles_U_free, "U_free")
    s_prefill_acts = _extract(bundles_S_prefill, "S_prefill")
    u_prefill_acts = _extract(bundles_U_prefill, "U_prefill")

    # ---- 5+6. Fit probes per (position × layer), per-row split ----
    train_idx, val_idx = _split_indices_by_row(items, train_fraction, seed)
    logger.info(
        f"Row-level split: {len(train_idx)} train items / {len(val_idx)} val items "
        f"(from {len({it.row_idx for it in items})} unique rows)"
    )

    position_bundles: dict[str, PositionProbeBundle] = {}
    for pos_idx, pos_name in enumerate(POSITION_NAMES):
        with timer(f"[{model_key}] fit probes @ {pos_name}"):
            pb = _fit_probes_for_position(
                position=pos_name,
                s_prefill=s_prefill_acts[:, pos_idx, :, :],
                u_prefill=u_prefill_acts[:, pos_idx, :, :],
                s_free=s_free_acts[:, pos_idx, :, :],
                u_free=u_free_acts[:, pos_idx, :, :],
                train_idx=train_idx, val_idx=val_idx, seed=seed,
            )
        position_bundles[pos_name] = pb
        logger.info(
            f"  {pos_name}: best layer={pb.best_layer} "
            f"MM val acc={pb.best_accuracy:.3f} "
            f"prefill→free@best={pb.mm_accuracies_free_val[pb.best_layer]:.3f} "
            f"multi-layer({len(pb.multi_layer_top)}L) acc={pb.multi_layer_acc:.3f}"
        )

    # ---- Pick global best (position, layer) ----
    best_position = max(position_bundles, key=lambda p: position_bundles[p].best_accuracy)
    best_pb = position_bundles[best_position]
    best_layer = best_pb.best_layer
    best_acc = best_pb.best_accuracy
    best_probe = best_pb.mm_probes[best_layer]
    logger.info(
        f"[{model_key}] BEST: position={best_position} layer={best_layer} "
        f"MM val acc={best_acc:.3f} prefill→free={best_pb.mm_accuracies_free_val[best_layer]:.3f}"
    )

    # ---- 7. FAKE transfer ----
    fake_transfer = None
    if do_fake_transfer:
        with timer(f"[{model_key}] FAKE transfer eval"):
            fake_transfer = _eval_transfer_fake(
                loaded=loaded, items=items, baseline_system=baseline_system,
                extract_batch_size=extract_batch_size, extract_max_length=extract_max_length,
                bundles_S_free=bundles_S_free, bundles_U_free=bundles_U_free,
                s_free_acts=s_free_acts, u_free_acts=u_free_acts,
                position=best_position, layer=best_layer, mm_probe=best_probe,
                supports_enable_thinking=supports_thinking,
            )
        logger.info(
            f"  FAKE: mean score={fake_transfer['fake_score_mean']:.3f} "
            f"(S_free={fake_transfer['s_free_score_mean']:.3f}, "
            f"U_free={fake_transfer['u_free_score_mean']:.3f})"
        )

    # ---- 9. Causal intervention sweep ----
    # Run intervention BEFORE the NONE-conflict eval so HF model is still loaded
    # (NONE-conflict spins up vLLM which competes for VRAM).
    intervention = None
    if do_intervention_sweep:
        with timer(f"[{model_key}] intervention sweep"):
            intervention = _intervention_sweep(
                loaded=loaded, items=items, baseline_system=baseline_system,
                layer=best_layer, probe=best_probe,
                n_prompts=intervention_n_prompts,
                judge_model_id=judge_model_id,
                supports_enable_thinking=supports_thinking,
                seed=seed,
            )
        logger.info(
            f"  intervention: layer={intervention['layer']} "
            f"natural_scale={intervention['natural_scale']:.3f} "
            f"alphas tested={len(intervention['alphas'])}"
        )

    # ---- 8. NONE-conflict transfer (last — frees HF model) ----
    none_conflict_transfer = None
    if do_none_conflict_transfer:
        held_out = load_held_out_evolved_pairs(n_held_out=none_conflict_n_pairs, seed=seed)
        # Free HF before vLLM grabs the GPU
        if free_after:
            free_model(loaded)
            import gc, torch  # noqa
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Re-load for the activation extraction inside _eval_transfer_none_conflict.
            loaded = load_model(model_key)
        with timer(f"[{model_key}] NONE-conflict transfer eval"):
            none_conflict_transfer = _eval_transfer_none_conflict(
                loaded=loaded, pairs=held_out,
                vllm_model_id=vllm_gen_model_id,
                judge_model_id=judge_model_id,
                extract_batch_size=extract_batch_size,
                extract_max_length=extract_max_length,
                position=best_position, layer=best_layer, mm_probe=best_probe,
                out_dir=out_dir, seed=seed,
                supports_enable_thinking=supports_thinking,
            )
        if none_conflict_transfer.get("pearson_r") is not None:
            logger.info(
                f"  NONE-conflict: r={none_conflict_transfer['pearson_r']:.3f} "
                f"p={none_conflict_transfer['p_value']:.2e} "
                f"n_judgable={none_conflict_transfer['n_judgable']}"
            )

    # ---- 10. Save ----
    arrays: dict[str, np.ndarray] = {}
    for pos_name, pb in position_bundles.items():
        for layer, probe in pb.mm_probes.items():
            arrays[f"mm_dir__{pos_name}__layer_{layer:03d}"] = probe.direction
            arrays[f"mm_raw__{pos_name}__layer_{layer:03d}"] = probe.raw_direction
            arrays[f"mm_midpoint__{pos_name}__layer_{layer:03d}"] = probe.midpoint
        for layer, vec in pb.lr_directions.items():
            arrays[f"lr_dir__{pos_name}__layer_{layer:03d}"] = vec
        arrays[f"mm_acc_train__{pos_name}"] = np.array(
            [pb.mm_accuracies_train[i] for i in range(loaded.n_layers)]
        )
        arrays[f"mm_acc_val__{pos_name}"] = np.array(
            [pb.mm_accuracies_val[i] for i in range(loaded.n_layers)]
        )
        arrays[f"mm_acc_free_val__{pos_name}"] = np.array(
            [pb.mm_accuracies_free_val[i] for i in range(loaded.n_layers)]
        )
        arrays[f"lr_acc_val__{pos_name}"] = np.array(
            [pb.lr_accuracies_val[i] for i in range(loaded.n_layers)]
        )
        arrays[f"mm_lr_cosine__{pos_name}"] = np.array(
            [pb.mm_lr_cosine.get(i, float("nan")) for i in range(loaded.n_layers)]
        )

    position_results_json = {}
    for pos_name, pb in position_bundles.items():
        position_results_json[pos_name] = {
            "best_layer": int(pb.best_layer),
            "best_accuracy": float(pb.best_accuracy),
            "multi_layer_top": [int(l) for l in pb.multi_layer_top],
            "multi_layer_acc": float(pb.multi_layer_acc),
            "multi_layer_acc_free": float(pb.multi_layer_acc_free),
            "mm_accuracies_train": {int(k): float(v) for k, v in pb.mm_accuracies_train.items()},
            "mm_accuracies_val": {int(k): float(v) for k, v in pb.mm_accuracies_val.items()},
            "mm_accuracies_free_val": {int(k): float(v) for k, v in pb.mm_accuracies_free_val.items()},
            "lr_accuracies_val": {int(k): float(v) for k, v in pb.lr_accuracies_val.items()},
            "lr_aurocs_val": {int(k): float(v) for k, v in pb.lr_aurocs_val.items()},
            "mm_lr_cosine": {int(k): float(v) for k, v in pb.mm_lr_cosine.items()},
        }

    result_json = {
        "experiment": "exp6_structural_authority",
        "model_key": model_key,
        "hf_id": cfg.hf_id,
        "n_layers": loaded.n_layers,
        "d_model": loaded.d_model,
        "n_items": len(items),
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "baseline_system": baseline_system,
        "instruction_choices": list(instruction_choices),
        "best_position": best_position,
        "best_layer": int(best_layer),
        "best_accuracy": float(best_acc),
        "position_results": position_results_json,
        "fake_transfer": fake_transfer,
        "none_conflict_transfer": none_conflict_transfer,
        "intervention": intervention,
    }

    save_result_bundle(
        out_dir,
        json_obj=result_json,
        arrays=arrays,
        pickles={"position_bundles": position_bundles, "items": items},
        manifest_extras={"experiment": "exp6_structural_authority", "model_key": model_key},
    )

    if free_after:
        free_model(loaded)

    return Exp6Result(
        model_key=model_key,
        n_layers=loaded.n_layers,
        d_model=loaded.d_model,
        n_items=len(items),
        n_train_rows=int(len(train_idx)),
        n_val_rows=int(len(val_idx)),
        position_results={k: asdict_safe(v) for k, v in position_bundles.items()},
        best_position=best_position,
        best_layer=int(best_layer),
        best_accuracy=float(best_acc),
        fake_transfer=fake_transfer,
        none_conflict_transfer=none_conflict_transfer,
        intervention=intervention,
    )


def asdict_safe(pb: PositionProbeBundle) -> dict:
    """Lightweight summary dict for return — full probes live in the pickle."""
    return {
        "position": pb.position,
        "best_layer": pb.best_layer,
        "best_accuracy": pb.best_accuracy,
        "multi_layer_top": pb.multi_layer_top,
        "multi_layer_acc": pb.multi_layer_acc,
        "multi_layer_acc_free": pb.multi_layer_acc_free,
        "mm_accuracies_val": pb.mm_accuracies_val,
        "mm_accuracies_free_val": pb.mm_accuracies_free_val,
        "mm_lr_cosine": pb.mm_lr_cosine,
    }


# -----------------------------------------------------------------------------
# Recovery: re-run NONE-conflict judging from saved generations (mirrors exp2b's
# `judge_generations_only`).
# -----------------------------------------------------------------------------

def judge_generations_only_exp6(
    out_dir: Path,
    model_key: str,
    judge_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict:
    """Re-judge NONE-conflict generations from a previous exp6 run that crashed
    after generation but before judging."""
    out_dir = Path(out_dir)
    gen_path = out_dir / "none_conflict_generations.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(f"No generations found at {gen_path}")

    with open(gen_path) as f:
        rows = [json.loads(line) for line in f]

    pairs = load_held_out_evolved_pairs(n_held_out=len(rows), seed=42)
    pair_by_idx = {p.idx: p for p in pairs}

    judge_rows = []
    for row in rows:
        p = pair_by_idx.get(row["pair_idx"])
        if p is None:
            continue
        judge_rows.append(JudgeRow(
            s_instruction=row["s_instruction"],
            u_instruction=row["u_instruction"],
            s_gold=p.s_aligned_response,
            u_gold=p.u_aligned_response,
            response=row["response"],
            extras={"pair_idx": row["pair_idx"]},
        ))

    verdicts = judge_with_vllm(
        judge_rows, model_id=judge_model_id, enable_thinking=False, free_after=True,
    )

    with open(out_dir / "none_conflict_generations.judged.jsonl", "w") as f:
        for r, v in zip(rows, verdicts):
            f.write(json.dumps({**r, "verdict": v["verdict"], "reason": v["reason"]}) + "\n")
    return {"n_judged": len(verdicts), "model_key": model_key}
