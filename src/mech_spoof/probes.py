"""Authority-direction probe training + difference-in-means.

Both methods return per-layer unit direction vectors. Compare via cosine similarity — if the
methods agree, the direction is robust; if they diverge, the probe is picking up on structure
beyond first-order mean difference (§4.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


@dataclass
class ProbeResult:
    probes: dict[int, LogisticRegression]
    accuracies: dict[int, float]
    aurocs: dict[int, float]
    directions: dict[int, np.ndarray]   # unit vectors, shape (d_model,)
    test_indices: np.ndarray
    labels_test: np.ndarray


def _normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + 1e-8)


def train_authority_probe(
    system_acts: np.ndarray,   # (n_S, n_layers, d_model)
    user_acts: np.ndarray,     # (n_U, n_layers, d_model)
    test_size: float = 0.25,
    C: float = 1.0,
    seed: int = 42,
    normalize: bool = True,
) -> ProbeResult:
    """Train per-layer logistic probes: label 1 = system, 0 = user.

    Returns a `ProbeResult` with per-layer probes, test accuracy, AUROC, and unit direction.
    """
    assert system_acts.ndim == 3 and user_acts.ndim == 3
    assert system_acts.shape[1:] == user_acts.shape[1:]
    n_layers, d_model = system_acts.shape[1], system_acts.shape[2]

    X_all = np.concatenate([system_acts, user_acts], axis=0)  # (n, n_layers, d_model)
    y_all = np.concatenate([
        np.ones(len(system_acts), dtype=int),
        np.zeros(len(user_acts), dtype=int),
    ])

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y_all)), y_all))

    probes: dict[int, LogisticRegression] = {}
    accuracies: dict[int, float] = {}
    aurocs: dict[int, float] = {}
    directions: dict[int, np.ndarray] = {}

    for layer in range(n_layers):
        X = X_all[:, layer, :]  # (n, d_model)
        if normalize:
            X = _normalize(X, axis=-1)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        probe = LogisticRegression(
            max_iter=1000, C=C, solver="lbfgs", random_state=seed
        )
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        y_prob = probe.predict_proba(X_test)[:, 1]

        accuracies[layer] = float(accuracy_score(y_test, y_pred))
        try:
            aurocs[layer] = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            aurocs[layer] = float("nan")

        w = probe.coef_[0]
        directions[layer] = w / (np.linalg.norm(w) + 1e-8)
        probes[layer] = probe

    return ProbeResult(
        probes=probes,
        accuracies=accuracies,
        aurocs=aurocs,
        directions=directions,
        test_indices=test_idx,
        labels_test=y_all[test_idx],
    )


def compute_authority_direction_dim(
    system_acts: np.ndarray,
    user_acts: np.ndarray,
) -> dict[int, np.ndarray]:
    """Difference-in-means direction per layer (Arditi et al. 2024 style)."""
    assert system_acts.shape[1:] == user_acts.shape[1:]
    n_layers = system_acts.shape[1]
    out: dict[int, np.ndarray] = {}
    for layer in range(n_layers):
        ms = system_acts[:, layer, :].mean(axis=0)
        mu = user_acts[:, layer, :].mean(axis=0)
        d = ms - mu
        out[layer] = d / (np.linalg.norm(d) + 1e-8)
    return out


def find_best_layer(accuracies: dict[int, float]) -> int:
    """Return layer index with highest probe accuracy. Ties: earliest."""
    return max(accuracies.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def score_activations(acts: np.ndarray, direction: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Project activations onto a unit direction. Returns one scalar per example."""
    if acts.ndim == 1:
        acts = acts[None, :]
    if normalize:
        acts = _normalize(acts, axis=-1)
    return acts @ direction


def cosine_agreement(
    dirs_a: dict[int, np.ndarray],
    dirs_b: dict[int, np.ndarray],
) -> dict[int, float]:
    """Per-layer cosine similarity between two direction sets (e.g. probe vs diff-in-means)."""
    out = {}
    for layer in dirs_a:
        if layer not in dirs_b:
            continue
        a, b = dirs_a[layer], dirs_b[layer]
        out[layer] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    return out
