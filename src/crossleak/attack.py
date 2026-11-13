"""Attacker decoders, labels and linkage for the main study (fitted in the analysis stage).

Features: the attacker's own forecast in model space (9 deciles x 24 h), the isolated forecast of its probe, calendar.
The no-channel control decoder sees only the isolated forecast and the calendar; the permuted control is trained on
shuffled labels. Labels and profile scores are defined in prereg/MAIN_PREREG.md section 4:

  L1 (primary)  the victim's previous-day evening level (z-profile, 18-23 h) is above the median of all users that day
                (within-day label: information shared by the whole population cannot predict it). Training labels use
                the median of the users the attacker can access (aux and public pools) on that day instead
  L2            the same level above the attacker's training median (the E0 label)
  L3            the victim's previous-day mean is below the 25th percentile of its own previous 13 daily means
  L4            the previous-day evening score is above the median of the victim's own previous 13 days (self-relative)
  L5            the self-relative residual of L4 is above the median residual of all users that day (an idiosyncratic
                event: neither the victim's stable routine nor a population-wide shock predicts it)
  R1 (primary)  within-day anomaly correlation: corr(yhat - mean_d, y - mean_d), mean_d = mean z-profile of all users
                that day
  R2            anomaly correlation against the training mean profile (the E0 metric)
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from crossleak.decode import evening_score
from crossleak.metrics import rowwise_pearson

RIDGE_ALPHAS = np.logspace(-2, 7, 19)
LOGIT_CS = np.logspace(-4, 2, 8)
LABELS = ("L1", "L2", "L3", "L4", "L5")


def labels(Y: np.ndarray, dm: np.ndarray | None, day: np.ndarray, ref_even_med: np.ndarray, train_median: float,
           ev14: np.ndarray | None = None, ref_resid_med: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Labels of the victim's previous day; -1 marks a sample where a label is undefined (excluded from that label)."""
    ev = evening_score(Y)
    r1 = ref_even_med[day]
    out = {"L1": np.where(np.isfinite(r1), (ev > r1).astype(int), -1), "L2": (ev > train_median).astype(int)}
    if dm is not None:
        prev = dm[:, :-1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            q25 = np.nanquantile(prev, 0.25, axis=1)
        out["L3"] = np.where(np.isfinite(q25) & np.isfinite(dm[:, -1]), (dm[:, -1] < q25).astype(int), -1)
    if ev14 is not None:
        prev = ev14[:, :-1]
        cnt = np.isfinite(prev).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(prev, axis=1)
        resid = ev14[:, -1] - med
        ok = (cnt >= 7) & np.isfinite(resid)
        out["L4"] = np.where(ok, (resid > 0).astype(int), -1)
        if ref_resid_med is not None:
            ok5 = ok & np.isfinite(ref_resid_med[day])
            out["L5"] = np.where(ok5, (resid > ref_resid_med[day]).astype(int), -1)
    return out


def features(arr: dict, prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    fxl = np.asarray(arr[f"fxl{prefix}"], dtype=np.float32)
    fiso = np.asarray(arr[f"fiso{prefix}"], dtype=np.float32)
    cal = np.asarray(arr[f"cal{prefix}"], dtype=np.float32)
    return np.hstack([fxl, fiso, cal]), np.hstack([fiso, cal])


def _logit(A, y, B, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        m = LogisticRegressionCV(Cs=LOGIT_CS, cv=3, max_iter=2000, scoring="roc_auc", random_state=seed).fit(A, y)
    return m.decision_function(B)


def linear_decoder(Xtr, Ytr, Ltr: dict, Xte, seed: int) -> dict:
    """Ridge reconstruction of the z-profile and one logistic score per label."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(A, Ytr)
    out = {"yhat": ridge.predict(B), "ridge_alpha": float(ridge.alpha_), "scores": {}}
    for name, y in Ltr.items():
        ok = y >= 0
        if len(np.unique(y[ok])) < 2:
            continue
        out["scores"][name] = _logit(A[ok], y[ok], B, seed)
    return out


def nonlinear_decoder(Xtr, Ytr, Ltr: dict, Xte, seed: int, labels_used=("L1",)) -> dict:
    """Gradient-boosted trees for labels, a one-hidden-layer MLP for the profile."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPRegressor

    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=500, random_state=seed).fit(A, Ytr)
    out = {"yhat": mlp.predict(B), "scores": {}}
    for name in labels_used:
        y = Ltr[name]
        ok = y >= 0
        if len(np.unique(y[ok])) < 2:
            continue
        hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, validation_fraction=0.15, random_state=seed)
        out["scores"][name] = hgb.fit(Xtr[ok], y[ok]).decision_function(Xte)
    return out


def within_day_r(yhat: np.ndarray, Y: np.ndarray, ref_mean_day: np.ndarray) -> np.ndarray:
    return rowwise_pearson(yhat - ref_mean_day, Y - ref_mean_day)


def linkage(yhat: np.ndarray, victims: np.ndarray, days: np.ndarray, cand: np.ndarray, pool: np.ndarray, N: int, seed: int) -> dict:
    """A1: rank the true co-tenant among N candidates by the correlation of the reconstructed previous-day profile
    with each candidate's old weekday profile (cand[u, d], from history 3-10 weeks before). Candidates are victim-pool
    users with an old profile; the N-1 decoys are drawn at random. Returns per-sample top-1 / top-5 hits (NaN if fewer
    than N candidates exist)."""
    rng = np.random.default_rng(seed)
    top1 = np.full(len(victims), np.nan)
    top5 = np.full(len(victims), np.nan)
    for i, (v, d) in enumerate(zip(victims, days)):
        if not np.isfinite(cand[v, d]).all():
            continue
        ok = pool[np.isfinite(cand[pool, d]).all(axis=1)]
        ok = ok[ok != v]
        if len(ok) < N - 1:
            continue
        cs = np.concatenate([[v], rng.choice(ok, size=N - 1, replace=False)])
        prof = cand[cs, d].astype(np.float64)
        pbar = prof.mean(axis=0, keepdims=True)
        s = rowwise_pearson(np.repeat((yhat[i] - pbar[0])[None], N, axis=0), prof - pbar)
        s = np.nan_to_num(s, nan=-2.0)
        rank = 1 + int((s[1:] > s[0]).sum())
        top1[i] = float(rank == 1)
        top5[i] = float(rank <= 5)
    return {"top1": top1, "top5": top5}
