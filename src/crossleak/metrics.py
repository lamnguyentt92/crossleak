"""Forecast accuracy, leakage metrics and cluster bootstrap."""

from __future__ import annotations

from typing import Callable

import numpy as np

DECILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def wql_item(y: np.ndarray, qpred: np.ndarray, qs: np.ndarray = DECILES) -> float:
    """Weighted quantile loss of one item: mean_q 2*sum_t pinball_q / sum_t |y| (NaN targets ignored).

    qpred: (Q, H) quantile forecasts in original units.
    """
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(y)
    if m.sum() == 0:
        return float("nan")
    e = y[None, m] - np.asarray(qpred, dtype=np.float64)[:, m]
    q = qs[:, None]
    pin = np.maximum(q * e, (q - 1.0) * e)
    den = np.abs(y[m]).sum()
    if den <= 0:
        return float("nan")
    return float((2.0 * pin.sum(axis=1)).mean() / den)


def wql_point(y: np.ndarray, point: np.ndarray) -> float:
    """WQL of a point forecast used for every quantile (equals sum|e| / sum|y| for symmetric deciles)."""
    return wql_item(y, np.repeat(np.asarray(point, dtype=np.float64)[None, :], len(DECILES), axis=0))


def mase_item(y: np.ndarray, median: np.ndarray, history: np.ndarray, season: int = 24) -> float:
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(history, dtype=np.float64)
    m = np.isfinite(y)
    d = np.abs(h[season:] - h[:-season])
    d = d[np.isfinite(d)]
    if m.sum() == 0 or len(d) == 0 or d.mean() == 0:
        return float("nan")
    return float(np.abs(y[m] - np.asarray(median)[m]).mean() / d.mean())


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(den > 0, r, np.nan)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (ties averaged). Returns NaN if only one class."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """Percentile CI of stat(indices) when resampling whole clusters with replacement."""
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    est = float(stat(np.arange(len(clusters))))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[c] for c in pick])
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"est": est, "lo": float("nan"), "hi": float("nan"), "n_clusters": int(len(uniq)), "n_boot_ok": 0}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"est": est, "lo": float(lo), "hi": float(hi), "n_clusters": int(len(uniq)), "n_boot_ok": len(vals)}
