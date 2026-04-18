"""Conflicting instructions dataset (§3.2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mech_spoof.configs import DATA_DIR
from mech_spoof.templates.base import PromptBundle, TemplateAdapter
from mech_spoof.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ConflictPair:
    id: str
    category: str
    system: str
    user: str
    eval: str
    system_wins: str
    user_wins: str


@dataclass
class ConflictPrompt:
    pair: ConflictPair
    real: PromptBundle   # system in real system field
    none: PromptBundle   # both in user field, no system
    fake: PromptBundle   # fake delimiter injection inside user field


def load_conflict_pairs(
    data_dir: Path = DATA_DIR,
    expanded_filename: str = "conflicting_pairs.json",
    seed_filename: str = "conflicting_pairs_seed.json",
) -> list[ConflictPair]:
    """Prefer the fully-expanded file; fall back to the seed file."""
    full = data_dir / expanded_filename
    seed = data_dir / seed_filename
    path = full if full.exists() else seed
    if not path.exists():
        raise FileNotFoundError(f"No conflict-pairs file at {full} or {seed}")
    with open(path) as f:
        raw = json.load(f)
    pairs = [ConflictPair(**item) for item in raw]
    logger.info(f"Loaded {len(pairs)} conflict pairs from {path.name}")
    return pairs


def build_conflicting_pairs(
    template: TemplateAdapter,
    pairs: list[ConflictPair] | None = None,
) -> list[ConflictPrompt]:
    """Build the REAL/NONE/FAKE prompts for every conflict pair."""
    if pairs is None:
        pairs = load_conflict_pairs()
    out: list[ConflictPrompt] = []
    for p in pairs:
        out.append(
            ConflictPrompt(
                pair=p,
                real=template.make_conflict_prompt(p.system, p.user, "REAL"),
                none=template.make_conflict_prompt(p.system, p.user, "NONE"),
                fake=template.make_conflict_prompt(p.system, p.user, "FAKE"),
            )
        )
    return out
