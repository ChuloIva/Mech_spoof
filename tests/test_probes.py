"""Probe unit test: plant a direction in synthetic Gaussians, probe recovers it."""

from __future__ import annotations

import numpy as np

from mech_spoof.probes import (
    compute_authority_direction_dim,
    cosine_agreement,
    find_best_layer,
    train_authority_probe,
)


def _synthetic(n_layers: int = 4, d_model: int = 64, n_per_class: int = 80,
               sep: float = 2.5, seed: int = 0):
    """Create (n_per_class, n_layers, d_model) arrays with a planted direction per layer.

    Each layer has its own direction. The "system" class is shifted along it by +sep, "user" by -sep.
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_layers, d_model))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sys_acts = np.zeros((n_per_class, n_layers, d_model))
    user_acts = np.zeros((n_per_class, n_layers, d_model))

    for layer in range(n_layers):
        noise_s = rng.normal(size=(n_per_class, d_model)) * 0.5
        noise_u = rng.normal(size=(n_per_class, d_model)) * 0.5
        sys_acts[:, layer, :] = sep * directions[layer] + noise_s
        user_acts[:, layer, :] = -sep * directions[layer] + noise_u
    return sys_acts, user_acts, directions


def test_probe_recovers_planted_direction():
    sys_acts, user_acts, planted = _synthetic()
    result = train_authority_probe(sys_acts, user_acts, seed=0)

    # All layers should have high accuracy given the strong planted signal
    for layer, acc in result.accuracies.items():
        assert acc > 0.9, f"layer {layer} accuracy {acc} below 0.9"

    # Recovered direction should align with planted (up to sign — probe direction points to sys=1)
    for layer in range(planted.shape[0]):
        cos = float(result.directions[layer] @ (planted[layer] /
                                                np.linalg.norm(planted[layer])))
        assert cos > 0.9, f"layer {layer} cosine {cos} below 0.9"


def test_dim_and_probe_agree():
    sys_acts, user_acts, _ = _synthetic()
    probe_res = train_authority_probe(sys_acts, user_acts, seed=0)
    dim_dirs = compute_authority_direction_dim(sys_acts, user_acts)
    agree = cosine_agreement(probe_res.directions, dim_dirs)
    for layer, c in agree.items():
        assert c > 0.8, f"layer {layer} probe-vs-dim cosine {c} below 0.8"


def test_find_best_layer_picks_maximum():
    accs = {0: 0.6, 1: 0.9, 2: 0.85, 3: 0.9}
    # Two layers tied at 0.9 — function should pick the earliest
    assert find_best_layer(accs) == 1
