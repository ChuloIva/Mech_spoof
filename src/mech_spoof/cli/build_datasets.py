"""Build or refresh the committed dataset JSONs via the Claude API.

Reads seed data, calls Claude to expand each category to the target count, writes
`data/structural/instructions_<cat>.json` and `data/conflicting_pairs.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mech_spoof.configs import (
    CONFLICT_CATEGORIES,
    CONFLICT_PAIRS_COUNT,
    DATA_DIR,
    STRUCTURAL_CATEGORIES,
    STRUCTURAL_N_PER_CATEGORY,
)
from mech_spoof.datasets._claude_gen import generate_conflict_pairs, generate_instructions
from mech_spoof.datasets.advbench import load_advbench
from mech_spoof.datasets.harmless import write_harmless_json
from mech_spoof.datasets.structural import _load_seeds
from mech_spoof.utils import get_logger

logger = get_logger(__name__)


def _expand_structural(target_per_category: int, model: str):
    seeds = _load_seeds(DATA_DIR)
    out_dir = DATA_DIR / "structural"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in STRUCTURAL_CATEGORIES:
        have = seeds.get(cat, [])
        need = max(0, target_per_category - len(have))
        if need == 0:
            items = have[:target_per_category]
        else:
            new = generate_instructions(cat, have, n=need, model=model)
            items = list(dict.fromkeys(have + new))[:target_per_category]
        out_path = out_dir / f"instructions_{cat}.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        logger.info(f"[{cat}] wrote {len(items)} items -> {out_path}")


def _expand_conflicts(target_count: int, model: str):
    seed_path = DATA_DIR / "conflicting_pairs_seed.json"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing {seed_path}")
    with open(seed_path) as f:
        seeds = json.load(f)

    by_cat: dict[str, list[dict]] = {}
    for p in seeds:
        by_cat.setdefault(p["category"], []).append(p)

    allowed_evals = [
        "language_check", "length_check", "format_check", "topic_check",
        "name_check", "tone_check", "string_check",
    ]
    eval_for_cat = {
        "language": "language_check",
        "length": "length_check",
        "format": "format_check",
        "topic": "topic_check",
        "name": "name_check",
        "tone": "tone_check",
        "string": "string_check",
    }

    out_pairs: list[dict] = []
    for cat, target in CONFLICT_CATEGORIES.items():
        have = by_cat.get(cat, [])
        need = max(0, target - len(have))
        if need == 0:
            out_pairs.extend(have[:target])
            continue
        new = generate_conflict_pairs(
            category=cat, eval_type=eval_for_cat[cat],
            allowed_evals=allowed_evals, examples=have, n=need, model=model,
        )
        # Assign ids if missing
        for i, p in enumerate(new, start=len(have) + 1):
            p.setdefault("id", f"{cat[:4]}_{i:02d}")
            p.setdefault("category", cat)
            p.setdefault("eval", eval_for_cat[cat])
        combined = have + new
        out_pairs.extend(combined[:target])

    out_path = DATA_DIR / "conflicting_pairs.json"
    with open(out_path, "w") as f:
        json.dump(out_pairs, f, indent=2)
    logger.info(f"wrote {len(out_pairs)} conflict pairs -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structural-n", type=int, default=STRUCTURAL_N_PER_CATEGORY)
    parser.add_argument("--conflict-n", type=int, default=CONFLICT_PAIRS_COUNT)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--skip-conflicts", action="store_true")
    parser.add_argument("--skip-advbench", action="store_true")
    args = parser.parse_args(argv)

    if not args.skip_structural:
        _expand_structural(args.structural_n, args.model)

    if not args.skip_conflicts:
        _expand_conflicts(args.conflict_n, args.model)

    if not args.skip_advbench:
        goals = load_advbench()
        logger.info(f"AdvBench: {len(goals)} harmful goals cached.")

    harmless_path = write_harmless_json(DATA_DIR)
    logger.info(f"harmless set -> {harmless_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
