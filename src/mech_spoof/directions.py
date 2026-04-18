"""Refusal-direction computation and authority-vs-refusal geometry (§6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mech_spoof.activations import extract_at_positions, response_first_position
from mech_spoof.models import LoadedModel
from mech_spoof.templates.base import PromptBundle


def _prompts_as_user(template, instructions: list[str]) -> list[PromptBundle]:
    """Wrap instructions as plain user prompts."""
    return [template.make_user_prompt(ins, user_followup="") for ins in instructions]


def compute_refusal_direction(
    loaded: LoadedModel,
    harmful: list[str],
    harmless: list[str],
    cache_dir=None,
) -> dict[int, np.ndarray]:
    """Difference-in-means refusal direction: mean(harmful) - mean(harmless) per layer."""
    template = loaded.template
    harmful_bundles = _prompts_as_user(template, harmful)
    harmless_bundles = _prompts_as_user(template, harmless)

    harmful_acts = extract_at_positions(
        loaded, harmful_bundles, response_first_position, cache_dir=cache_dir
    )
    harmless_acts = extract_at_positions(
        loaded, harmless_bundles, response_first_position, cache_dir=cache_dir
    )

    out: dict[int, np.ndarray] = {}
    n_layers = harmful_acts.shape[1]
    for layer in range(n_layers):
        mh = harmful_acts[:, layer, :].mean(axis=0)
        ml = harmless_acts[:, layer, :].mean(axis=0)
        d = mh - ml
        out[layer] = d / (np.linalg.norm(d) + 1e-8)
    return out


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
    """Project x onto unit direction. Returns a vector of the same shape as x."""
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    coeff = x @ direction
    if x.ndim == 1:
        return coeff * direction
    return coeff[:, None] * direction[None, :]


def reject(x: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove the `direction` component from x."""
    return x - project(x, direction)
