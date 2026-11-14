"""v0.4 causal-defence study: the public context (D1) and the public centres of DP-Prototype (D4) restricted to data
from before the forecast origin.

The frozen main driver (``experiments/main.py``, hash unchanged) took D1 windows at least ``min_gap_days`` away from the
origin in *either* direction and fitted the LifeSnaps D4 centres on public users from any day. A deployment can use only
past data. This driver reuses the main driver unchanged and adds, on the same cold-start targets:

  d1p_<k>            k - 1 public windows of the same weekday at least ``min_gap_days`` days *before* the origin
  d4p_P<P>_eps<e>    D4 whose public centres are fitted only on public users' contexts ending before the origin
                     (only for datasets without a training period; ELD's centres already come from 2013)

next to the original ``d1_<k>`` and ``d4_P<P>_eps<e>`` (paired estimate of the effect of causality). Only utility cells
run. :func:`install` points the main module's ``build_cells``, ``Env`` and utility runner at this module; the main
module's source is not modified.
"""

from __future__ import annotations

import logging

import numpy as np

from crossleak.config import stable_seed
from crossleak.defenses import dp_prototype as dp
from crossleak.experiments import main as M
from crossleak.metrics import wql_item
from crossleak.model import pad_left

log = logging.getLogger("crossleak.v4")

_ORIGINAL = {"build_cells": M.build_cells, "Env": M.Env, "util": M.RUNNERS["util"]}


def d1p_name(k: int) -> str:
    return f"d1p_{k}"


def d4p_name(P: int, e) -> str:
    return "d4p" + M.d4_name(P, e)[2:]


def past_centres_datasets(cfg: dict) -> list[str]:
    return [n for n, d in cfg["datasets"].items() if not d.get("train_days")]


def build_cells(cfg: dict) -> list:
    """Only the utility cells of the configuration (the main builder also adds ref/dpaudit cells per dataset)."""
    return [c for c in _ORIGINAL["build_cells"](cfg) if c.exp == "util"]


class CausalEnv(_ORIGINAL["Env"]):
    """Main environment plus past-only public centres and releases for datasets without a training period."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self._centres_past: dict = {}
        self._releases_past: dict = {}

    def public_past_days(self, name: str, d: int) -> np.ndarray:
        ds = self.ds(name)
        return np.where(ds.split["any"] & (ds.count("public") > 0) & (np.arange(ds.D) < d))[0]

    def centres_past(self, name: str, P: int, d: int) -> np.ndarray | None:
        """k-means on public users' representations whose 14-day contexts end before origin d (None if < 2P)."""
        key = (name, P, d)
        if key not in self._centres_past:
            ds = self.ds(name)
            kind = self.cfg["d4"]["representation"]
            days = self.public_past_days(name, d)
            out = None
            if len(days):
                rng = np.random.default_rng(stable_seed("centres_past", name, str(d), base=self.cfg["seed"]))
                days = rng.choice(days, size=min(len(days), int(self.cfg["d4"]["centre_days"])), replace=False)
                R = [dp.represent(ds.ctx(int(u), int(dd)), kind) for dd in days for u in ds.users("public", int(dd))]
                if len(R) >= 2 * P:
                    out = dp.fit_centres(np.stack(R), P, stable_seed("kmeans_past", name, str(P), str(d), base=self.cfg["seed"]))
            self._centres_past[key] = out
        return self._centres_past[key]

    def release_past(self, name: str, d: int, P: int, eps: float) -> dp.Release | None:
        key = (name, d, P, eps)
        if key not in self._releases_past:
            c = self.centres_past(name, P, d)
            rel = None
            if c is not None:
                ds = self.ds(name)
                kind = self.cfg["d4"]["representation"]
                users = ds.users("service", d)
                R = np.stack([dp.represent(ds.ctx(int(u), d), kind) for u in users]) if len(users) else np.zeros((0, dp.dim_of(kind, ds.C)))
                rng = np.random.default_rng(stable_seed("release", name, str(d), str(P), str(eps), base=self.cfg["seed"]))
                rel = dp.release(R, c, eps, float(self.cfg["delta"]), dp.default_clip(kind, ds.C), rng, float(self.cfg["d4"]["sum_share"]))
            self._releases_past[key] = rel
        return self._releases_past[key]


def _public_windows(ds, pub: np.ndarray, pub_valid: np.ndarray, d: int, gap: int, past_only: bool) -> tuple[list[int], np.ndarray]:
    dist = d - np.arange(ds.D)
    ok = (ds.weekday == ds.weekday[d]) & ((dist >= gap) if past_only else (np.abs(dist) >= gap))
    ok_days = np.where(ok)[0]
    return [i for i in range(len(pub)) if pub_valid[i, ok_days].any()], ok_days


def run_util(cell, env: CausalEnv, seed: int) -> tuple[dict, dict]:
    """The main utility runner plus d1p and d4p configurations on the same targets (paired)."""
    cfg, p = env.cfg, cell.p
    ds = env.ds(cell.dataset)
    C, L = ds.C, p["ell"]
    xl, d1 = list(p["xl"]), list(p["d1"])
    kmax = max(xl) if xl else 1
    rng = np.random.default_rng(seed)
    gap = int(cfg["d1"]["min_gap_days"])
    pub = ds.pools["public"]
    pub_valid = ds.valid[pub] & ds.split["any"][None, :]
    past_c = cell.dataset in past_centres_datasets(cfg)
    days = ds.feasible("test", {"target": 1, "service": kmax - 1})
    need = max(d1) - 1 if d1 else 0
    if d1:  # origins with enough past public windows (the past-only rule is the binding one)
        days = np.array([d for d in days if len(_public_windows(ds, pub, pub_valid, int(d), gap, True)[0]) >= need], dtype=int)
    if p["d4"] and past_c:
        days = np.array([d for d in days if env.centres_past(cell.dataset, cfg["d4"]["P"][0], int(d)) is not None], dtype=int)
    if len(days) == 0:
        return M._infeasible("no test origin with past public data, a target and enough established users")
    configs: list[str] = ["iso", "full"] + [f"xl{k}" for k in xl] + [f"d1_{k}" for k in d1] + [d1p_name(k) for k in d1]
    if p["d4"]:
        configs += [M.d4_name(P, e) for P in cfg["d4"]["P"] for e in cfg["d4"]["eps"]]
        if past_c:
            configs += [d4p_name(P, e) for P in cfg["d4"]["P"] for e in cfg["d4"]["eps"]]
    kind = cfg["d4"]["representation"]
    groups, owner, targets = [], [], []
    tries = 0
    while len(targets) < p["n"]:
        tries += 1
        if tries > 50 * p["n"]:
            break
        d = M.draw_day(rng, days)
        u = M.pick(rng, ds, "target", d, 1, None)
        cot = M.pick(rng, ds, "service", d, kmax - 1, None)
        if u is None or cot is None:
            continue
        u = int(u[0])
        y = ds.hor(u, d)
        if not np.isfinite(y).any() or np.nansum(np.abs(y)) <= 0:
            continue
        full = ds.ctx(u, d)
        tgt = pad_left(full[-L:], C)
        rows = {"iso": [tgt], "full": [full]}
        for k in xl:
            rows[f"xl{k}"] = [tgt] + [ds.ctx(int(c), d) for c in cot[: k - 1]]
        if d1:
            for past_only, namef in ((False, lambda k: f"d1_{k}"), (True, d1p_name)):
                cand, ok_days = _public_windows(ds, pub, pub_valid, d, gap, past_only)
                chosen = rng.choice(cand, size=need, replace=False)
                wins = []
                for i in chosen:
                    dd = ok_days[pub_valid[i, ok_days]]
                    wins.append(ds.ctx(int(pub[i]), int(rng.choice(dd))))
                for k in d1:
                    rows[namef(k)] = [tgt] + wins[: k - 1]
        if p["d4"]:
            for P in cfg["d4"]["P"]:
                for e in cfg["d4"]["eps"]:
                    rel = env.release(cell.dataset, d, P, float(e))
                    rows[M.d4_name(P, e)] = [tgt] + [dp.to_series(r, kind, C) for r in rel.prototypes]
                    if past_c:
                        relp = env.release_past(cell.dataset, d, P, float(e))
                        rows[d4p_name(P, e)] = [tgt] + [dp.to_series(r, kind, C) for r in relp.prototypes]
        for c in configs:
            groups.append(np.stack(rows[c]))
            owner.append((len(targets), c))
        targets.append((u, d, y, full))
    if not targets:
        return M._infeasible("no target drawn")
    fc = env.forecaster(cell.model)
    preds = fc.predict_groups(groups)
    W = {c: np.full(len(targets), np.nan) for c in configs}
    for pr, (i, c) in zip(preds, owner):
        W[c][i] = wql_item(targets[i][2], pr[0])
    snaive = np.array([wql_item(t[2], np.repeat(t[3][-24:][None], 9, axis=0)) for t in targets])
    arr = {f"wql_{c}": W[c] for c in configs}
    arr.update({"user": np.array([t[0] for t in targets]), "day": np.array([t[1] for t in targets]), "wql_snaive": snaive})
    iso = W["iso"]
    summary = {"n_targets": len(targets), "n_users": int(len(np.unique(arr["user"]))), "configs": configs,
               "n_feasible_days": int(len(days)), "past_centres": bool(past_c),
               "wql_mean": {c: float(np.nanmean(W[c])) for c in configs},
               "gain_vs_iso": {c: float(1 - np.nanmean(W[c]) / np.nanmean(iso)) for c in configs}}
    return summary, arr


def install() -> None:
    """Route the main driver's cell list, environment and utility runner through this module (idempotent)."""
    M.build_cells = build_cells
    M.Env = CausalEnv
    M.RUNNERS["util"] = run_util


def uninstall() -> None:
    M.build_cells = _ORIGINAL["build_cells"]
    M.Env = _ORIGINAL["Env"]
    M.RUNNERS["util"] = _ORIGINAL["util"]
