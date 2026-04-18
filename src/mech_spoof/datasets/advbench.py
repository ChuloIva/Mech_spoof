"""AdvBench harmful-instructions loader (Zou et al. 2023).

Downloads the `harmful_behaviors.csv` once, caches as JSON. Used by Exp 3 (refusal direction)
and Exp 4 (attack payloads).
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import requests

from mech_spoof.configs import DATA_DIR
from mech_spoof.utils import get_logger

logger = get_logger(__name__)

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
    "main/data/advbench/harmful_behaviors.csv"
)


def load_advbench(cache_dir: Path = DATA_DIR, refresh: bool = False) -> list[str]:
    """Return the list of AdvBench harmful goals (strings)."""
    cache = cache_dir / "advbench_harmful.json"
    if cache.exists() and not refresh:
        with open(cache) as f:
            return json.load(f)

    logger.info(f"Downloading AdvBench from {ADVBENCH_URL}...")
    resp = requests.get(ADVBENCH_URL, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    goals = [row["goal"].strip() for row in reader if row.get("goal")]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(goals, f, indent=2)
    logger.info(f"Cached {len(goals)} AdvBench goals to {cache}")
    return goals
