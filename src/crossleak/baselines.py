"""Lightweight forecasting baselines for short histories (v0.5; requested in the manuscript review).

A global ridge forecaster maps the last 24 hours (instance-scaled), a missing-value mask and the weekday of the target
day to the next 24 hours; probabilistic forecasts add per-step residual quantiles estimated on a held-out quarter of
the training *users*. Training windows are distinct (user, day) pairs. It is trained only on users that cold-start targets never belong to (auxiliary and public pools)
and, where a dataset has a training period (ELD: 2013), only on that period.

Two instance scalings are implemented: ``meanabs`` (divide by the mean absolute value of the last 24 hours) and
``meanstd`` (subtract the mean, divide by the standard deviation). :func:`fit_ridge` with ``scaling="select"`` fits
both and keeps the one with the lower WQL on the held-out training users (never on evaluation targets).
"""

from __future__ import annotations

import numpy as np

from crossleak.metrics import wql_item
from crossleak.model import DECILES

SCALINGS = ("meanabs", "meanstd")


def scale_of(last24: np.ndarray, scaling: str) -> tuple[float, float]:
    x = np.asarray(last24, dtype=np.float64)
    if scaling == "meanabs":
        s = float(np.nanmean(np.abs(x)))
        return 0.0, (s if np.isfinite(s) and s > 1e-6 else 1.0)
    if scaling == "meanstd":
        m, s = float(np.nanmean(x)), float(np.nanstd(x))
        if not np.isfinite(m):
            m = 0.0
        if not (np.isfinite(s) and s > 1e-6):
            s = max(abs(m), 1.0) * 1e-3 + 1e-6
        return m, s
    raise ValueError(scaling)


def features(last24: np.ndarray, weekday: int, scaling: str) -> tuple[np.ndarray, float, float]:
    m, s = scale_of(last24, scaling)
    z = (np.asarray(last24, dtype=np.float64) - m) / s
    mask = np.isfinite(z).astype(np.float32)
    z = np.where(np.isfinite(z), z, 0.0)
    wd = np.zeros(7, np.float32)
    wd[int(weekday)] = 1.0
    return np.concatenate([z, mask, wd]).astype(np.float32), m, s


def training_windows(ds, n_max: int, seed: int, before_day: int | None = None) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    """Distinct (user, day) windows of auxiliary and public users: (last 24 h, next 24 h, weekday of the target day,
    user). Training split if the dataset has one, any day otherwise; with ``before_day``, only windows whose next 24 h
    end before that origin. At most ``n_max`` windows, drawn without replacement."""
    rng = np.random.default_rng(seed)
    pools = np.concatenate([ds.pools["aux"], ds.pools["public"]])
    split = ds.split["train"] if ds.cfg["datasets"][ds.name].get("train_days") else ds.split["any"]
    ok = ds.valid[pools] & split[None, :]
    if before_day is not None:
        ok[:, before_day:] = False
    pairs = np.argwhere(ok)
    pairs = pairs[rng.permutation(len(pairs))]
    out = []
    for i, d in pairs:
        if len(out) >= n_max:
            break
        u = int(pools[i])
        last = ds.ctx(u, int(d))[-24:].astype(np.float64)
        y = ds.hor(u, int(d)).astype(np.float64)
        if before_day is not None and ds.val.origins[int(d)] + ds.H > ds.val.origins[before_day]:
            continue
        if np.isfinite(last).sum() < 12 or not np.isfinite(y).all():
            continue
        out.append((last, y, int(ds.weekday[int(d)]), u))
    return out


def _fit_one(win: list, scaling: str, seed: int) -> dict:
    """Ridge on 3/4 of the training *users*; residual quantiles and the held-out WQL on the other quarter."""
    from sklearn.linear_model import RidgeCV

    X, Y, MS = [], [], []
    for last, y, wd, _u in win:
        f, m, s = features(last, wd, scaling)
        X.append(f)
        Y.append(np.clip((y - m) / s, -50, 50))
        MS.append((m, s))
    X, Y = np.stack(X), np.stack(Y)
    users = np.array([w[3] for w in win])
    uniq = np.unique(users)
    held = np.random.default_rng(seed).permutation(uniq)[: max(1, len(uniq) // 4)]
    cal = np.flatnonzero(np.isin(users, held))
    fit = np.flatnonzero(~np.isin(users, held))
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[fit], Y[fit])
    pred_cal = model.predict(X[cal])
    qres = np.quantile(Y[cal] - pred_cal, DECILES, axis=0)
    wql = [wql_item(win[i][1], (pred_cal[j][None, :] + qres) * MS[i][1] + MS[i][0]) for j, i in enumerate(cal)]
    return {"model": model, "qres": qres, "scaling": scaling, "alpha": float(model.alpha_), "n_fit": int(len(fit)),
            "n_cal": int(len(cal)), "n_unique": int(len(win)), "n_users": int(len(uniq)), "n_users_cal": int(len(held)),
            "cal_wql": float(np.nanmean(wql))}


def fit_ridge(ds, n_max: int = 20000, seed: int = 11, scaling: str = "select", before_day: int | None = None) -> dict:
    win = training_windows(ds, n_max, seed, before_day)
    if len(win) < 40:
        raise ValueError(f"only {len(win)} training windows")
    if scaling != "select":
        return _fit_one(win, scaling, seed)
    fits = {s: _fit_one(win, s, seed) for s in SCALINGS}
    best = min(fits.values(), key=lambda r: r["cal_wql"])
    best = dict(best)
    best["selection"] = {s: r["cal_wql"] for s, r in fits.items()}
    best["alternatives"] = fits
    return best


def predict_ridge(rm: dict, last24: np.ndarray, weekday: int) -> np.ndarray:
    """(9, 24) deciles in original units."""
    f, m, s = features(last24, weekday, rm["scaling"])
    point = rm["model"].predict(f[None])[0]
    return (point[None, :] + rm["qres"]) * s + m
