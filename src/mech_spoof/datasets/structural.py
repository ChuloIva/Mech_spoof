"""Structural contrastive dataset builder (§3.1).

Generates 100 (or N) instructions per category, wraps each in Condition S and Condition U
using the model's template, splits train/test.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from mech_spoof.configs import (
    DATA_DIR,
    STRUCTURAL_CATEGORIES,
    STRUCTURAL_N_PER_CATEGORY,
    STRUCTURAL_TRAIN_FRACTION,
)
from mech_spoof.templates.base import PromptBundle, TemplateAdapter
from mech_spoof.utils import get_logger, set_seed

logger = get_logger(__name__)


@dataclass
class StructuralDataset:
    instructions: list[str]
    categories: list[str]               # per-instruction category
    prompts_system: list[PromptBundle]  # Condition S
    prompts_user: list[PromptBundle]    # Condition U
    train_idx: list[int]
    test_idx: list[int]


def load_structural_instructions(
    data_dir: Path = DATA_DIR,
    n_per_category: int = STRUCTURAL_N_PER_CATEGORY,
) -> dict[str, list[str]]:
    """Load instructions per category. Prefers `structural/instructions_{cat}.json` (full expansion),
    falls back to `structural/seeds.json` for seed data."""
    out: dict[str, list[str]] = {}
    full_dir = data_dir / "structural"
    for cat in STRUCTURAL_CATEGORIES:
        full_path = full_dir / f"instructions_{cat}.json"
        if full_path.exists():
            with open(full_path) as f:
                items = json.load(f)
            out[cat] = items[:n_per_category]
            continue
    if out:
        missing = [c for c in STRUCTURAL_CATEGORIES if c not in out]
        if missing:
            seeds = _load_seeds(data_dir)
            for cat in missing:
                out[cat] = seeds.get(cat, [])
        return out

    seeds = _load_seeds(data_dir)
    for cat in STRUCTURAL_CATEGORIES:
        out[cat] = seeds.get(cat, [])[:n_per_category]
    return out


def _load_seeds(data_dir: Path) -> dict[str, list[str]]:
    path = data_dir / "structural" / "seeds.json"
    if not path.exists():
        raise FileNotFoundError(f"Structural seeds file missing: {path}")
    with open(path) as f:
        return json.load(f)


def build_structural_contrastive(
    template: TemplateAdapter,
    instructions_by_category: dict[str, list[str]] | None = None,
    train_fraction: float = STRUCTURAL_TRAIN_FRACTION,
    seed: int = 42,
) -> StructuralDataset:
    """Build Condition S and Condition U prompts for every instruction.

    Returns a dataset with per-instruction prompt bundles and a stable train/test split.
    """
    if instructions_by_category is None:
        instructions_by_category = load_structural_instructions()

    all_instructions: list[str] = []
    categories: list[str] = []
    for cat in STRUCTURAL_CATEGORIES:
        items = instructions_by_category.get(cat, [])
        all_instructions.extend(items)
        categories.extend([cat] * len(items))

    if not all_instructions:
        raise ValueError("No instructions loaded — check data/structural/seeds.json")

    logger.info(
        f"Building structural dataset: {len(all_instructions)} instructions across "
        f"{len(set(categories))} categories"
    )

    prompts_system = [template.make_system_prompt(ins) for ins in all_instructions]
    prompts_user = [template.make_user_prompt(ins) for ins in all_instructions]

    set_seed(seed)
    n = len(all_instructions)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cutoff = int(n * train_fraction)
    train_idx = sorted(idx[:cutoff])
    test_idx = sorted(idx[cutoff:])

    return StructuralDataset(
        instructions=all_instructions,
        categories=categories,
        prompts_system=prompts_system,
        prompts_user=prompts_user,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def build_structural_contrastive_matched(
    template: TemplateAdapter,
    baseline_system: str = "You are a helpful assistant.",
    instructions_by_category: dict[str, list[str]] | None = None,
    train_fraction: float = STRUCTURAL_TRAIN_FRACTION,
    seed: int = 42,
) -> StructuralDataset:
    """Control dataset where both conditions share an identical baseline system block.

    Only the *placement* of the instruction-of-interest varies:
      - S: system = "{baseline}\\n{instruction}", user = followup
      - U: system = "{baseline}",                  user = "{instruction}\\n\\n{followup}"

    Kills the trivial "is there a system block?" feature that plain S vs U leaks.
    """
    if instructions_by_category is None:
        instructions_by_category = load_structural_instructions()

    all_instructions: list[str] = []
    categories: list[str] = []
    for cat in STRUCTURAL_CATEGORIES:
        items = instructions_by_category.get(cat, [])
        all_instructions.extend(items)
        categories.extend([cat] * len(items))

    if not all_instructions:
        raise ValueError("No instructions loaded — check data/structural/seeds.json")

    logger.info(
        f"Building structural MATCHED-baseline dataset: {len(all_instructions)} instructions "
        f"across {len(set(categories))} categories (baseline={baseline_system!r})"
    )

    prompts_system = [
        template.make_system_prompt_matched(ins, baseline_system=baseline_system)
        for ins in all_instructions
    ]
    prompts_user = [
        template.make_user_prompt_matched(ins, baseline_system=baseline_system)
        for ins in all_instructions
    ]

    set_seed(seed)
    n = len(all_instructions)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cutoff = int(n * train_fraction)
    train_idx = sorted(idx[:cutoff])
    test_idx = sorted(idx[cutoff:])

    return StructuralDataset(
        instructions=all_instructions,
        categories=categories,
        prompts_system=prompts_system,
        prompts_user=prompts_user,
        train_idx=train_idx,
        test_idx=test_idx,
    )
