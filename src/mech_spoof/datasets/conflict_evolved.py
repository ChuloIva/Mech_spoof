"""Load evolved conflict pairs and build 4-trace contrastive bundles for Exp 1b.

Each conflict pair produces 4 PromptBundles (system-following × 2 orderings + user-following × 2),
with gold responses appended as assistant content. The role swap deconfounds instruction content
from the authority label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mech_spoof.configs import DATA_DIR
from mech_spoof.templates.base import PromptBundle
from mech_spoof.utils import get_logger

logger = get_logger(__name__)

DEFAULT_PATH = DATA_DIR / "conflict_pairs_stratified_1k.jsonl"


@dataclass
class EvolvedConflictPair:
    idx: int
    s_instruction: str
    u_instruction: str
    s_aligned_response: str
    u_aligned_response: str
    conflict_axis: str
    macro_axis: str


@dataclass
class ConflictTraceQuad:
    pair: EvolvedConflictPair
    sys_follow_ab: PromptBundle   # sys=A, user=B, resp=A-aligned  (+)
    usr_follow_ab: PromptBundle   # sys=A, user=B, resp=B-aligned  (−)
    sys_follow_ba: PromptBundle   # sys=B, user=A, resp=B-aligned  (+)
    usr_follow_ba: PromptBundle   # sys=B, user=A, resp=A-aligned  (−)


def load_evolved_conflict_pairs(
    path: Path | None = None,
    max_pairs: int | None = None,
) -> list[EvolvedConflictPair]:
    path = Path(path) if path else DEFAULT_PATH
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_pairs and i >= max_pairs:
                break
            row = json.loads(line)
            pairs.append(EvolvedConflictPair(
                idx=i,
                s_instruction=row["s_instruction"],
                u_instruction=row["u_instruction"],
                s_aligned_response=row["s_aligned_response"],
                u_aligned_response=row["u_aligned_response"],
                conflict_axis=row["conflict_axis"],
                macro_axis=row.get("macro_axis", "unknown"),
            ))
    logger.info(f"Loaded {len(pairs)} evolved conflict pairs from {path}")
    return pairs


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


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


def _make_bundle(
    tokenizer,
    sys_instr: str,
    user_instr: str,
    response: str,
    condition: str,
    extras: dict[str, Any],
    supports_enable_thinking: bool = False,
    system_role_supported: bool = True,
) -> PromptBundle:
    if system_role_supported:
        messages_full = [
            {"role": "system", "content": sys_instr},
            {"role": "user", "content": user_instr},
            {"role": "assistant", "content": response},
        ]
        messages_prefix = messages_full[:-1]
    else:
        messages_full = [
            {"role": "user", "content": f"{sys_instr}\n\n{user_instr}"},
            {"role": "assistant", "content": response},
        ]
        messages_prefix = messages_full[:-1]

    extra_kwargs = {"enable_thinking": False} if supports_enable_thinking else {}

    text = tokenizer.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False, **extra_kwargs,
    )
    full_ids = _flatten_ids(tokenizer.apply_chat_template(
        messages_full, tokenize=True, add_generation_prompt=False, **extra_kwargs,
    ))
    # Prefix with generation prompt: ends exactly where the assistant turn would begin generating.
    prefix_ids = _flatten_ids(tokenizer.apply_chat_template(
        messages_prefix, tokenize=True, add_generation_prompt=True, **extra_kwargs,
    ))

    response_start = min(len(prefix_ids), len(full_ids))
    response_end = len(full_ids)
    # Sanity fallback if template does something unexpected.
    if response_end <= response_start:
        response_start = max(0, response_end - 1)

    extras_with_span = dict(extras)
    extras_with_span["response_token_span"] = (response_start, response_end)

    return PromptBundle(
        text=text,
        input_ids=full_ids,
        instruction_text=response[:100],
        instruction_token_span=(0, response_start),
        response_first_pos=response_start,
        condition=condition,
        extras=extras_with_span,
    )


def build_conflict_traces(
    tokenizer,
    pairs: list[EvolvedConflictPair],
    supports_enable_thinking: bool = False,
    system_role_supported: bool = True,
    max_response_chars: int = 3000,
) -> list[ConflictTraceQuad]:
    quads = []
    for pair in pairs:
        a = pair.s_instruction
        b = pair.u_instruction
        resp_a = _truncate(pair.s_aligned_response, max_response_chars)
        resp_b = _truncate(pair.u_aligned_response, max_response_chars)
        common = dict(
            tokenizer=tokenizer,
            supports_enable_thinking=supports_enable_thinking,
            system_role_supported=system_role_supported,
        )
        base_extras = {
            "pair_idx": pair.idx,
            "macro_axis": pair.macro_axis,
            "conflict_axis": pair.conflict_axis,
        }

        # Trace 1: sys=A, user=B, resp=A-aligned → system-following
        t1 = _make_bundle(
            sys_instr=a, user_instr=b, response=resp_a, condition="S",
            extras={**base_extras, "trace": 1}, **common,
        )
        # Trace 2: sys=A, user=B, resp=B-aligned → user-following
        t2 = _make_bundle(
            sys_instr=a, user_instr=b, response=resp_b, condition="U",
            extras={**base_extras, "trace": 2}, **common,
        )
        # Trace 3: sys=B, user=A, resp=B-aligned → system-following
        t3 = _make_bundle(
            sys_instr=b, user_instr=a, response=resp_b, condition="S",
            extras={**base_extras, "trace": 3}, **common,
        )
        # Trace 4: sys=B, user=A, resp=A-aligned → user-following
        t4 = _make_bundle(
            sys_instr=b, user_instr=a, response=resp_a, condition="U",
            extras={**base_extras, "trace": 4}, **common,
        )
        quads.append(ConflictTraceQuad(
            pair=pair,
            sys_follow_ab=t1,
            usr_follow_ab=t2,
            sys_follow_ba=t3,
            usr_follow_ba=t4,
        ))
    logger.info(f"Built {len(quads)} trace quads ({len(quads) * 4} total bundles)")
    return quads


def flatten_traces(
    quads: list[ConflictTraceQuad],
) -> tuple[list[PromptBundle], list[PromptBundle]]:
    sys_following: list[PromptBundle] = []
    usr_following: list[PromptBundle] = []
    for q in quads:
        sys_following.append(q.sys_follow_ab)
        sys_following.append(q.sys_follow_ba)
        usr_following.append(q.usr_follow_ab)
        usr_following.append(q.usr_follow_ba)
    return sys_following, usr_following
