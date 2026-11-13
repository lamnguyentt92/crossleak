"""Attacker decoders and their controls (shared by the experiment runner and the re-analysis).

The attacker observes its own forecast (in model space) and can recompute the isolated
forecast of its own probe with the public model. The *no-channel* control attacker gets
the isolated forecast and calendar features only. Leakage = metric(channel) - metric(control).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.preprocessing import StandardScaler

from crossleak.metrics import auc, cluster_bootstrap, rowwise_pearson

RIDGE_ALPHAS = np.logspace(-2, 7, 19)
LOGIT_CS = np.logspace(-4, 2, 10)


def zprofile(last24: np.ndarray) -> np.ndarray:
    """Shape of the victim's last day: linearly interpolated over gaps, then z-scored."""
    x = np.asarray(last24, dtype=np.float64)
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.full_like(x, np.nan)
    if not m.all():
        idx = np.arange(len(x))
        x = np.interp(idx, idx[m], x[m])
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def evening_score(z: np.ndarray) -> np.ndarray:
    """Mean normalised level over 18:00-23:00 of the label day (z rows of length 24)."""
    return np.asarray(z)[:, 18:24].mean(axis=1)


def _fit_predict(Xtr, Ytr, Ltr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(A, Ytr)
    yhat = ridge.predict(B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        logit = LogisticRegressionCV(Cs=LOGIT_CS, cv=5, max_iter=3000, scoring="roc_auc", random_state=seed).fit(A, Ltr)
    score = logit.decision_function(B)
    return yhat, score, float(ridge.alpha_)


def decode_eval(
    Xtr_xl: np.ndarray,
    Xtr_ctrl: np.ndarray,
    Ytr: np.ndarray,
    Str: np.ndarray,
    Xte_xl: np.ndarray,
    Xte_ctrl: np.ndarray,
    Yte: np.ndarray,
    Ste: np.ndarray,
    clusters_te: np.ndarray,
    seed: int,
    n_boot: int,
) -> dict:
    """Profile reconstruction (ridge) and binary evening-label inference (logistic).

    Y*: (n, 24) z-profiles; S*: (n,) evening scores; the label threshold is the training median.
    Profile accuracy is the *anomaly correlation*: Pearson r between prediction and truth after
    subtracting the training mean profile, so that the daily shape shared by everybody (which any
    attacker can guess) does not count as leakage. Raw r is reported as a secondary number.
    """
    thr = float(np.median(Str))
    Ltr, Lte = (Str > thr).astype(int), (Ste > thr).astype(int)
    yx, sx, ax = _fit_predict(Xtr_xl, Ytr, Ltr, Xte_xl, seed)
    yc, scr, ac = _fit_predict(Xtr_ctrl, Ytr, Ltr, Xte_ctrl, seed)
    perm = np.random.default_rng(seed).permutation(len(Ytr))
    yp, sp, _ = _fit_predict(Xtr_xl, Ytr[perm], Ltr[perm], Xte_xl, seed)
    ybar = Ytr.mean(axis=0, keepdims=True)
    r_x, r_c, r_p = (rowwise_pearson(a - ybar, Yte - ybar) for a in (yx, yc, yp))
    raw_x, raw_c = rowwise_pearson(yx, Yte), rowwise_pearson(yc, Yte)
    ok = np.isfinite(r_x) & np.isfinite(r_c)
    cl = clusters_te[ok]
    rx, rc = r_x[ok], r_c[ok]
    d_r = cluster_bootstrap(lambda i: float(np.mean(rx[i] - rc[i])), cl, n_boot, seed)
    d_auc = cluster_bootstrap(lambda i: auc(Lte[i], sx[i]) - auc(Lte[i], scr[i]), clusters_te, n_boot, seed + 1)
    auc_x = cluster_bootstrap(lambda i: auc(Lte[i], sx[i]), clusters_te, n_boot, seed + 2)
    return {
        "label_threshold": thr,
        "label_rate_test": float(Lte.mean()),
        "ridge_alpha_xl": ax,
        "ridge_alpha_ctrl": ac,
        "r_median_xl": float(np.nanmedian(r_x)),
        "r_median_ctrl": float(np.nanmedian(r_c)),
        "r_median_perm": float(np.nanmedian(r_p)),
        "r_raw_median_xl": float(np.nanmedian(raw_x)),
        "r_raw_median_ctrl": float(np.nanmedian(raw_c)),
        "delta_r": d_r,
        "auc_xl": auc_x,
        "auc_ctrl": auc(Lte, scr),
        "auc_perm": auc(Lte, sp),
        "delta_auc": d_auc,
        "n_test_valid": int(ok.sum()),
    }
