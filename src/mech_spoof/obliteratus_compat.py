"""Bridge to the OBLITERATUS abliteration library (third_party/OBLITERATUS).

OBLITERATUS is AGPL-3.0; dynamic imports from it may make this combined program a covered
work under AGPL. Treat carefully if you plan to redistribute.

We use only the parts that line up with our refusal-direction methodology:
  - The bundled 512-pair harmful/harmless prompt corpus, plus the AdvBench / HarmBench / HH-RLHF
    / WildJailbreak loaders registered in `obliteratus.prompts.DATASET_SOURCES`.
  - The "knee_cosmic" layer-selection heuristic (elbow on norm curve + bottom-10% cosine
    similarity between harmful and harmless means).

Both re-implementations of the selection heuristics live here so the rest of our code doesn't
import obliteratus directly.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

from mech_spoof.utils import get_logger

logger = get_logger(__name__)


# ---------- Import plumbing ----------

_THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "OBLITERATUS"


def _ensure_on_path() -> None:
    """Put the in-repo OBLITERATUS checkout on sys.path so `import obliteratus` works
    even when the third_party submodule isn't pip-installed."""
    override = os.environ.get("OBLITERATUS_PATH")
    path = Path(override) if override else _THIRD_PARTY
    if not path.exists():
        raise ImportError(
            f"OBLITERATUS checkout not found at {path}. "
            "Either clone it into third_party/OBLITERATUS or set OBLITERATUS_PATH."
        )
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def available() -> bool:
    """True if OBLITERATUS can be imported without errors."""
    try:
        _ensure_on_path()
        import obliteratus.prompts  # noqa: F401
        return True
    except Exception as e:
        logger.debug(f"obliteratus not importable: {e}")
        return False


# ---------- Prompt dataset access ----------

VALID_SOURCES = ("builtin", "advbench", "harmbench", "anthropic_redteam", "wildjailbreak")


def load_prompt_pairs(source: str = "builtin") -> tuple[list[str], list[str]]:
    """Load a (harmful, harmless) pair of prompt lists from OBLITERATUS.

    Valid sources:
        "builtin"           — 512 hand-curated pairs shipped with OBLITERATUS
        "advbench"          — Zou et al. 2023 (~520)
        "harmbench"         — Mazeika et al. 2024 (~510)
        "anthropic_redteam" — Anthropic HH-RLHF red-team (~2000)
        "wildjailbreak"     — Jiang et al. 2024 (~2000 paired)
    """
    if source not in VALID_SOURCES:
        raise ValueError(f"Unknown source {source!r}. Valid: {VALID_SOURCES}")
    _ensure_on_path()
    from obliteratus.prompts import load_dataset_source
    harmful, harmless = load_dataset_source(source)
    logger.info(
        f"[obliteratus] loaded source={source}: {len(harmful)} harmful / {len(harmless)} harmless"
    )
    return harmful, harmless


def load_builtin_prompt_pairs() -> tuple[list[str], list[str]]:
    """Shortcut: OBLITERATUS's 512 curated built-in pairs."""
    return load_prompt_pairs("builtin")


# ---------- Layer-selection heuristics (replicated, not imported) ----------

def knee_layer_selection(
    layer_norms: dict[int, float],
    threshold_fraction: float = 0.05,
) -> list[int]:
    """Kneedle-style elbow on the sorted direction-norm curve.

    Replicates `_select_layers_knee` in obliteratus/abliterate.py. Given
    `layer_norms = {layer_idx: ||refusal_direction||}`, returns layers from the top down
    to the knee, filtered to at least `threshold_fraction * max_norm`.
    """
    if not layer_norms:
        return []
    sorted_layers = sorted(layer_norms.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_layers) <= 2:
        return [idx for idx, _ in sorted_layers]

    norms = [n for _, n in sorted_layers]
    max_n = norms[0]
    if max_n == 0:
        return []
    normalized = [n / max_n for n in norms]

    n_pts = len(normalized)
    x_start, y_start = 0.0, normalized[0]
    x_end, y_end = 1.0, normalized[-1]
    line_len = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)

    best_dist = -1.0
    best_k = 1
    for i in range(1, n_pts - 1):
        x_i = i / (n_pts - 1)
        y_i = normalized[i]
        dist = abs(
            (y_end - y_start) * x_i
            - (x_end - x_start) * y_i
            + x_end * y_start
            - y_end * x_start
        ) / line_len
        if dist > best_dist:
            best_dist = dist
            best_k = i + 1

    min_threshold = max_n * threshold_fraction
    selected = [idx for idx, norm in sorted_layers[:best_k] if norm >= min_threshold]
    return selected if selected else [sorted_layers[0][0]]


def cosmic_layer_selection(
    harmful_means: dict[int, np.ndarray],
    harmless_means: dict[int, np.ndarray],
    bottom_fraction: float = 0.10,
) -> list[int]:
    """COSMIC layer selection (arXiv:2506.00085).

    Picks layers with the LOWEST cosine similarity between the harmful-mean and harmless-mean
    activations — these are the most separable layers, where refusal is most encoded.
    """
    if not harmful_means or not harmless_means:
        return []

    sims: list[tuple[int, float]] = []
    for idx, h in harmful_means.items():
        if idx not in harmless_means:
            continue
        s = harmless_means[idx]
        h_norm = np.linalg.norm(h)
        s_norm = np.linalg.norm(s)
        if h_norm < 1e-8 or s_norm < 1e-8:
            continue
        sims.append((idx, float(h @ s / (h_norm * s_norm))))

    if len(sims) < 3:
        return [idx for idx, _ in sims]

    sims.sort(key=lambda kv: kv[1])
    n_select = max(1, min(len(sims) // 2, int(len(sims) * bottom_fraction + 0.5)))
    return [idx for idx, _ in sims[:n_select]]


def knee_cosmic_layer_selection(
    layer_norms: dict[int, float],
    harmful_means: dict[int, np.ndarray],
    harmless_means: dict[int, np.ndarray],
) -> list[int]:
    """Union of knee + COSMIC — matches OBLITERATUS's `layer_selection="knee_cosmic"` default."""
    knee = set(knee_layer_selection(layer_norms))
    cosmic = set(cosmic_layer_selection(harmful_means, harmless_means))
    return sorted(knee | cosmic)
