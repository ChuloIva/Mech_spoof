"""Dataset schema tests."""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def test_structural_seeds_schema():
    with open(DATA / "structural" / "seeds.json") as f:
        seeds = json.load(f)
    for cat in ("format", "persona", "behavioral", "style"):
        assert cat in seeds, f"missing category {cat}"
        assert len(seeds[cat]) >= 15, f"category {cat} has too few seeds"
        for item in seeds[cat]:
            assert isinstance(item, str) and len(item) >= 5


def test_conflict_seeds_schema():
    with open(DATA / "conflicting_pairs_seed.json") as f:
        pairs = json.load(f)
    required_keys = {"id", "category", "system", "user", "eval", "system_wins", "user_wins"}
    allowed_evals = {
        "language_check", "length_check", "format_check", "topic_check",
        "name_check", "tone_check", "string_check",
    }
    for p in pairs:
        assert set(p.keys()) >= required_keys, f"missing keys in pair: {p}"
        assert p["eval"] in allowed_evals, f"unknown eval type {p['eval']}"


def test_attack_payload_builder_returns_text():
    from mech_spoof.datasets.attacks import build_attack_payloads

    class _FakeTemplate:
        name = "chatml"
    payloads = build_attack_payloads(_FakeTemplate(), ["demonstrate how to X"], max_per_attack=3)
    assert "chatinject_single_turn" in payloads
    for atype, items in payloads.items():
        for p in items:
            assert p.text and isinstance(p.text, str)
            assert p.harmful_goal == "demonstrate how to X"
