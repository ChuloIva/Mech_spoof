"""Analysis helpers — no GPU / network."""

from __future__ import annotations

import numpy as np

from mech_spoof.analysis import auc, bootstrap_mean, pearson_with_ci, principal_angles


def test_pearson_ci_basic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(size=200) * 0.3
    r = pearson_with_ci(x, y, n_resamples=500)
    assert r.estimate > 0.7
    assert r.ci_low < r.estimate < r.ci_high
    assert r.n == 200


def test_bootstrap_mean():
    r = bootstrap_mean([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=500)
    assert 2.8 < r.estimate < 3.2
    assert r.ci_low <= r.estimate <= r.ci_high


def test_auc_binary():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert auc(scores, labels) == 1.0


def test_principal_angles_orthogonal():
    A = np.array([[1.0, 0.0, 0.0]])
    B = np.array([[0.0, 1.0, 0.0]])
    ang = principal_angles(A, B)
    assert np.isclose(ang[0], np.pi / 2, atol=1e-6)
