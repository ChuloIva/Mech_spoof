"""Statistical helpers: correlation with CI, bootstrap, AUC, principal angles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap, pearsonr
from sklearn.metrics import roc_auc_score

from mech_spoof.configs import BOOTSTRAP_N_RESAMPLES


@dataclass
class StatResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float | None = None
    n: int | None = None


def pearson_with_ci(
    x, y, n_resamples: int = BOOTSTRAP_N_RESAMPLES, confidence: float = 0.95
) -> StatResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r, p = pearsonr(x, y)

    def stat(x_, y_):
        return pearsonr(x_, y_).statistic

    res = bootstrap(
        (x, y),
        statistic=stat,
        paired=True,
        vectorized=False,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="percentile",
    )
    return StatResult(
        estimate=float(r),
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        p_value=float(p),
        n=len(x),
    )


def bootstrap_mean(
    values, n_resamples: int = BOOTSTRAP_N_RESAMPLES, confidence: float = 0.95
) -> StatResult:
    values = np.asarray(values, dtype=float)
    res = bootstrap(
        (values,),
        statistic=np.mean,
        n_resamples=n_resamples,
        confidence_level=confidence,
        method="percentile",
    )
    return StatResult(
        estimate=float(values.mean()),
        ci_low=float(res.confidence_interval.low),
        ci_high=float(res.confidence_interval.high),
        p_value=None,
        n=len(values),
    )


def auc(scores, labels) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between row-spans of A and B. Each row is a direction."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)
