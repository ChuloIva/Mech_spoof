"""Refusal-direction computation and authority-vs-refusal geometry (§6).

Default refusal-direction method matches OBLITERATUS / Arditi et al.:
    direction = normalize(mean(harmful_acts) - mean(harmless_acts))
read at the LAST TOKEN of the RAW prompt (no chat template). This makes results directly
comparable with OBLITERATUS (third_party/OBLITERATUS/obliteratus/abliterate.py:1625). A
chat-template-wrapped alternative is available via `wrap_mode="chat"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from mech_spoof.activations import extract_at_positions, response_first_position
from mech_spoof.models import LoadedModel
from mech_spoof.templates.base import PromptBundle


WrapMode = Literal["raw", "chat"]


# ---------- Prompt wrapping ----------

def _raw_bundle(tokenizer, text: str) -> PromptBundle:
    """Tokenize a raw string with BOS/special tokens; residual is read at the last token.

    Matches OBLITERATUS's default (raw text, position -1).
    """
    ids = tokenizer.encode(text, add_special_tokens=True)
    return PromptBundle(
        text=text,
        input_ids=list(ids),
        instruction_text=text,
        instruction_token_span=(0, len(ids)),
        response_first_pos=len(ids) - 1,
        condition="U",
    )


def _prompts_as_user(template, instructions: list[str]) -> list[PromptBundle]:
    """Wrap instructions as plain user prompts via the model's chat template."""
    return [template.make_user_prompt(ins, user_followup="") for ins in instructions]


def _build_bundles(loaded: LoadedModel, texts: list[str], wrap_mode: WrapMode):
    if wrap_mode == "raw":
        return [_raw_bundle(loaded.tokenizer, t) for t in texts]
    if wrap_mode == "chat":
        return _prompts_as_user(loaded.template, texts)
    raise ValueError(f"Unknown wrap_mode {wrap_mode!r}")


# ---------- Refusal direction (returns per-layer means + directions) ----------


@dataclass
class RefusalResult:
    directions: dict[int, np.ndarray]           # unit vectors per layer
    norms: dict[int, float]                     # ||harmful_mean - harmless_mean|| per layer
    harmful_mean: dict[int, np.ndarray]         # per-layer harmful activation mean
    harmless_mean: dict[int, np.ndarray]        # per-layer harmless activation mean
    wrap_mode: WrapMode = "raw"
    source: str = "custom"
    strong_layers: list[int] = field(default_factory=list)   # knee_cosmic selection
    n_harmful: int = 0
    n_harmless: int = 0


def compute_refusal_direction(
    loaded: LoadedModel,
    harmful: list[str] | None = None,
    harmless: list[str] | None = None,
    wrap_mode: WrapMode = "raw",
    source: str | None = None,
    cache_dir: Path | None = None,
    select_strong_layers: bool = True,
) -> RefusalResult:
    """Compute per-layer refusal directions via difference-in-means.

    Parameters
    ----------
    harmful, harmless : list[str], optional
        Provide your own prompt lists. If omitted and `source` is given (or defaults to
        "builtin"), pulls from OBLITERATUS.
    wrap_mode : "raw" | "chat"
        "raw"  — tokenize as-is, read last token. OBLITERATUS default.
        "chat" — apply the model's chat template as a user turn. More realistic for instruct
                 models but not directly comparable with Arditi/OBLITERATUS results.
    source : str, optional
        One of "builtin" | "advbench" | "harmbench" | "anthropic_redteam" | "wildjailbreak".
        Ignored if `harmful` and `harmless` are both provided.
    select_strong_layers : bool
        If True, run OBLITERATUS's knee+COSMIC layer selection and populate
        `result.strong_layers`.
    """
    from mech_spoof.obliteratus_compat import knee_cosmic_layer_selection, load_prompt_pairs

    if harmful is None or harmless is None:
        source = source or "builtin"
        harmful_loaded, harmless_loaded = load_prompt_pairs(source)
        harmful = harmful or harmful_loaded
        harmless = harmless or harmless_loaded
    else:
        source = source or "custom"

    bundles_harmful = _build_bundles(loaded, list(harmful), wrap_mode)
    bundles_harmless = _build_bundles(loaded, list(harmless), wrap_mode)

    def positions_fn(_loaded, bundle):
        return bundle.response_first_pos

    cache_h = Path(cache_dir) / "harmful" if cache_dir else None
    cache_s = Path(cache_dir) / "harmless" if cache_dir else None

    harmful_acts = extract_at_positions(loaded, bundles_harmful, positions_fn, cache_dir=cache_h)
    harmless_acts = extract_at_positions(loaded, bundles_harmless, positions_fn, cache_dir=cache_s)

    n_layers = harmful_acts.shape[1]
    directions: dict[int, np.ndarray] = {}
    norms: dict[int, float] = {}
    h_means: dict[int, np.ndarray] = {}
    l_means: dict[int, np.ndarray] = {}

    for layer in range(n_layers):
        mh = harmful_acts[:, layer, :].mean(axis=0)
        ml = harmless_acts[:, layer, :].mean(axis=0)
        diff = mh - ml
        n = float(np.linalg.norm(diff))
        norms[layer] = n
        directions[layer] = diff / (n + 1e-8)
        h_means[layer] = mh
        l_means[layer] = ml

    strong: list[int] = []
    if select_strong_layers:
        strong = knee_cosmic_layer_selection(norms, h_means, l_means)

    return RefusalResult(
        directions=directions,
        norms=norms,
        harmful_mean=h_means,
        harmless_mean=l_means,
        wrap_mode=wrap_mode,
        source=source,
        strong_layers=strong,
        n_harmful=len(harmful),
        n_harmless=len(harmless),
    )


# ---------- Authority-refusal geometry ----------


@dataclass
class GeometryReport:
    cosine_by_layer: dict[int, float]
    principal_angle_deg_by_layer: dict[int, float]
    shared_variance_by_layer: dict[int, float]


def analyze_authority_refusal_relationship(
    authority_dirs: dict[int, np.ndarray],
    refusal_dirs: dict[int, np.ndarray],
) -> GeometryReport:
    """Per-layer cosine + principal angle between authority and refusal directions."""
    cos_map: dict[int, float] = {}
    angle_map: dict[int, float] = {}
    shared_map: dict[int, float] = {}
    for layer in authority_dirs:
        if layer not in refusal_dirs:
            continue
        a = authority_dirs[layer]
        r = refusal_dirs[layer]
        cos = float(a @ r / (np.linalg.norm(a) * np.linalg.norm(r) + 1e-8))
        cos_map[layer] = cos
        angle_map[layer] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        M = np.stack([a, r], axis=0)
        s = np.linalg.svd(M, compute_uv=False)
        shared_map[layer] = float(s[0] / (s[0] + s[1] + 1e-8))
    return GeometryReport(
        cosine_by_layer=cos_map,
        principal_angle_deg_by_layer=angle_map,
        shared_variance_by_layer=shared_map,
    )


def project(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project x onto unit direction."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    coeff = x @ direction
    if x.ndim == 1:
        return coeff * direction
    return coeff[:, None] * direction[None, :]


def reject(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the `direction` component from x."""
    return x - project(x, direction)
