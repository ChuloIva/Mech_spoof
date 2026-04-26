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


@dataclass
class DiffMeanProbe:
    """Mass-mean probe (Marks & Tegmark, COLM 2024) for one layer.

    Stores both the unit direction (for cosine comparison and projection scoring) and
    `raw_direction = mean_A - mean_B` so callers can pick a natural intervention scale.
    """

    direction: np.ndarray       # unit vector, shape (d_model,)
    midpoint: np.ndarray        # (mean_A + mean_B) / 2
    raw_direction: np.ndarray   # un-normalised mean_A - mean_B
    mean_A: np.ndarray
    mean_B: np.ndarray
    n_A: int
    n_B: int
    layer: int

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Project X (n, d) onto the direction, centred on the midpoint."""
        return (X - self.midpoint[np.newaxis, :]) @ self.direction

    def classify_batch(self, X: np.ndarray) -> np.ndarray:
        return (self.score_batch(X) > 0).astype(int)

    def accuracy(self, X: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.classify_batch(X) == labels))

    @property
    def natural_scale(self) -> float:
        """||mean_A - mean_B|| — intervention strength of "one full centroid shift"."""
        return float(np.linalg.norm(self.raw_direction))


def fit_diff_mean_probes(
    acts_A: np.ndarray,        # (n_A, n_layers, d_model)
    acts_B: np.ndarray,        # (n_B, n_layers, d_model)
) -> dict[int, DiffMeanProbe]:
    """Fit one MM probe per layer. No normalisation — matches the reference paper."""
    assert acts_A.ndim == 3 and acts_B.ndim == 3
    assert acts_A.shape[1:] == acts_B.shape[1:]
    n_layers = acts_A.shape[1]
    out: dict[int, DiffMeanProbe] = {}
    for layer in range(n_layers):
        X_A = acts_A[:, layer, :]
        X_B = acts_B[:, layer, :]
        mean_A = X_A.mean(axis=0)
        mean_B = X_B.mean(axis=0)
        raw = mean_A - mean_B
        norm = float(np.linalg.norm(raw))
        unit = raw / (norm + 1e-10) if norm > 1e-10 else np.zeros_like(raw)
        out[layer] = DiffMeanProbe(
            direction=unit,
            midpoint=(mean_A + mean_B) / 2,
            raw_direction=raw,
            mean_A=mean_A,
            mean_B=mean_B,
            n_A=acts_A.shape[0],
            n_B=acts_B.shape[0],
            layer=layer,
        )
    return out


def fit_diff_mean_multi_layer(
    acts_A: np.ndarray,
    acts_B: np.ndarray,
    layers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate activations across `layers` and fit one MM probe in the joint space.

    Returns (unit_direction, midpoint, raw_direction). Useful for picking up
    layer-disagreement structure that single-layer probes miss.
    """
    X_A = np.concatenate([acts_A[:, l, :] for l in layers], axis=1)
    X_B = np.concatenate([acts_B[:, l, :] for l in layers], axis=1)
    mean_A = X_A.mean(axis=0)
    mean_B = X_B.mean(axis=0)
    raw = mean_A - mean_B
    norm = float(np.linalg.norm(raw))
    unit = raw / (norm + 1e-10) if norm > 1e-10 else np.zeros_like(raw)
    midpoint = (mean_A + mean_B) / 2
    return unit, midpoint, raw


def score_multi_layer(
    acts: np.ndarray,        # (n, n_layers_total, d_model)
    direction: np.ndarray,
    midpoint: np.ndarray,
    layers: list[int],
) -> np.ndarray:
    X = np.concatenate([acts[:, l, :] for l in layers], axis=1)
    return (X - midpoint[np.newaxis, :]) @ direction


def intervene_along_direction(
    loaded,
    input_ids,
    direction: np.ndarray,
    layer: int,
    alpha: float,
    positions: list[int] | int | None = None,
    max_new_tokens: int = 128,
):
    """Generate from `loaded.hf_model` while adding `alpha * direction` to layer `layer`'s
    residual stream at the specified position(s).

    Parameters
    ----------
    loaded : LoadedModel
    input_ids : 1D or 2D long tensor / list
    direction : unit vector in residual space (shape d_model)
    layer : which transformer block to intervene at
    alpha : scalar — typically a multiple of probe.natural_scale
    positions : token positions to perturb. None → all positions in the prompt.
                During generation the cached positions list is used; new tokens
                are not perturbed (the intervention is applied only on the prompt
                pass).
    max_new_tokens : passed to model.generate

    Returns the generated token ids (including prompt).
    """
    import torch

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(loaded.device)
    prompt_len = int(input_ids.shape[1])

    direction_t = torch.tensor(direction, dtype=torch.float32, device=loaded.device)

    if isinstance(positions, int):
        positions = [positions]

    def _hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # h: (b, s, d). Only perturb the prompt forward pass; during incremental decode
        # the seq dim is 1 and we leave it untouched.
        if h.shape[1] != prompt_len:
            return out
        if positions is None:
            h = h + alpha * direction_t.to(h.dtype)
        else:
            h = h.clone()
            for p in positions:
                if 0 <= p < h.shape[1]:
                    h[:, p, :] = h[:, p, :] + alpha * direction_t.to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    module = loaded.layer_module(layer)
    handle = module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            gen = loaded.hf_model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=loaded.tokenizer.pad_token_id or loaded.tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    return gen


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
