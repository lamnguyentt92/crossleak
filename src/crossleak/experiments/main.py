"""Main study (v0.2): utility of cross-learning and defences, co-tenant attacks including sybils, channel strength
up to the default batch size, a DP audit of the D4 release, and edge cost.

Families of cells (all resumable, one directory per cell; large cells are split into *parts*):
  ref        per dataset, no model: day-level population references and old-history candidate profiles (A1)
  util       per dataset x model x history length: isolated, full context, cross-learning (XL), public context (D1),
             DP-Prototype (D4) on the same cold-start targets (matched)
  attack     per configuration x role (train/test) x part: the attacker's features and the victim's labels; decoders
             are fitted in the analysis stage (``analysis/main_report.py``)
  channel    per configuration x part: swap influence of one victim on the attacker's output
  invariance per dataset x model: exact positive-affine invariance
  dpaudit    per dataset, no model: label inference by an adversary that knows every other user (D4 release)
  cost       phase "cost", one idle machine: latency and throughput per mode

Seeds: crc32(sample key) XOR seed; the sample key excludes the model, so both models see identical samples.
"""

from __future__ import annotations

import json
import logging
import math
import resource
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crossleak.config import TAG, hostname, runs_dir, stable_seed
from crossleak.data.panel import compute_validity, day_mask, load_panel, make_pools
from crossleak.decode import evening_score, zprofile
from crossleak.defenses import dp_prototype as dp
from crossleak.metrics import wql_item
from crossleak.model import Forecaster, instance_loc_scale, pad_left

log = logging.getLogger("crossleak.main")

FAMILY_PRIORITY = {"ref": 0, "invariance": 0, "dpaudit": 0, "util24": 1, "attack_single2": 2, "attack_sybil": 3, "channel": 4,
                   "attack_single": 5, "util": 6, "attack_other": 7, "cost": 9}
SETUP_SEC = {"ref": 120.0, "dpaudit": 300.0}


# --------------------------------------------------------------------------- cells


def _v(v) -> str:
    """Parameter value as it appears in cell ids (no spaces or brackets)."""
    if isinstance(v, (tuple, list)):
        return "x".join(str(x) for x in v) if v else "none"
    return str(v)


@dataclass(frozen=True)
class Cell:
    exp: str
    dataset: str
    model: str
    params: tuple  # sorted (key, value) pairs

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def sample_key(self) -> str:
        return "__".join([self.exp, self.dataset] + [f"{k}-{_v(v)}" for k, v in self.params])

    @property
    def cell_id(self) -> str:
        return "__".join([self.exp, self.dataset, self.model] + [f"{k}-{_v(v)}" for k, v in self.params])

    @property
    def family_key(self) -> str:
        """Identifies the configuration across parts and roles (what the analysis merges)."""
        return "__".join([self.exp, self.dataset, self.model] + [f"{k}-{_v(v)}" for k, v in self.params if k not in ("part", "role", "n")])

    @property
    def priority(self) -> int:
        p = self.p
        if self.exp == "util":
            return FAMILY_PRIORITY["util24" if p["ell"] == 24 else "util"]
        if self.exp == "attack":
            if p["fam"] == "single":
                return FAMILY_PRIORITY["attack_single2" if p["k"] == 2 else "attack_single"]
            return FAMILY_PRIORITY["attack_sybil" if (p["fam"] == "sybil" and p["filler"] == "copy") else "attack_other"]
        return FAMILY_PRIORITY.get(self.exp, 8)

    def n_series(self, cfg: dict) -> int:
        """Number of forecast series this cell computes (planning only)."""
        p = self.p
        if self.exp == "util":
            per = 2 + sum(p["xl"]) + sum(p["d1"])
            if p["d4"]:
                per += len(cfg["d4"]["eps"]) * sum(P + 1 for P in cfg["d4"]["P"])
            return p["n"] * per
        if self.exp == "attack":
            return p["n"] * (p["k"] + 1)
        if self.exp == "channel":
            return p["n"] * (2 * p["k"] + 1)
        if self.exp == "invariance":
            return cfg["invariance"]["n"] * 5 * cfg["invariance"]["k"]
        return 0

    def cost(self, cfg: dict) -> float:
        """Estimated seconds on one busy testbed machine (planning and load balancing only)."""
        if self.exp in SETUP_SEC:
            return SETUP_SEC[self.exp]
        if self.exp == "cost":
            return 900.0
        return self.n_series(cfg) * float(cfg["cost_per_series"][self.model]) + 30.0


def _parts(n: int, per_series: int, sec: float, budget: float) -> list[int]:
    """Split n samples into parts that each cost <= budget seconds."""
    per_part = max(1, int(budget // max(per_series * sec, 1e-9)))
    sizes = []
    left = n
    while left > 0:
        sizes.append(min(per_part, left))
        left -= sizes[-1]
    return sizes


def build_cells(cfg: dict) -> list[Cell]:
    cells: list[Cell] = []
    budget = float(cfg["part_budget_sec"])
    cps = cfg["cost_per_series"]
    for ds in cfg["datasets"]:
        cells.append(Cell("ref", ds, "none", ()))
        cells.append(Cell("dpaudit", ds, "none", ()))
    inv = cfg["invariance"]
    for ds in inv["datasets"]:
        for m in inv["models"]:
            cells.append(Cell("invariance", ds, m, (("k", inv["k"]),)))
    nP = sum(P + 1 for P in cfg["d4"]["P"]) * len(cfg["d4"]["eps"])
    for u in cfg["utility"]:
        per = 2 + sum(u["xl_k"]) + sum(u["d1_k"]) + (nP if u["d4"] else 0)
        for m in u["models"]:
            for i, n in enumerate(_parts(u["n_targets"], per, cps[m], budget)):
                cells.append(Cell("util", u["dataset"], m, tuple(sorted({
                    "ell": u["ell"], "xl": tuple(u["xl_k"]), "d1": tuple(u["d1_k"]), "d4": bool(u["d4"]), "part": i, "n": n}.items()))))
    sizes = {int(k): v for k, v in cfg["attack"]["sizes"].items()}
    for f in cfg["attack"]["families"]:
        for ds in f["datasets"]:
            for m in f["models"]:
                for k in f["k"]:
                    for probe in f["probes"]:
                        for role, n_role in zip(("train", "test"), sizes[k]):
                            for i, n in enumerate(_parts(n_role, k + 1, cps[m], budget)):
                                cells.append(Cell("attack", ds, m, tuple(sorted({
                                    "fam": f["fam"], "filler": f.get("filler", "none"), "k": k, "probe": probe, "role": role, "part": i, "n": n}.items()))))
    npairs = {int(k): v for k, v in cfg["channel"]["n_pairs"].items()}
    for f in cfg["channel"]["families"]:
        for ds in f["datasets"]:
            for m in f["models"]:
                for k in f["k"]:
                    for probe in f["probes"]:
                        for i, n in enumerate(_parts(npairs[k], 2 * k + 1, cps[m], budget)):
                            cells.append(Cell("channel", ds, m, tuple(sorted({
                                "fam": f["fam"], "filler": f.get("filler", "none"), "k": k, "probe": probe, "part": i, "n": n}.items()))))
    for m in cfg["cost"]["models"]:
        cells.append(Cell("cost", "synthetic", m, ()))
    ids = [c.cell_id for c in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell ids in the configuration")
    return cells


def assign_shards(cells: list[Cell], cfg: dict, nshards: int) -> dict[str, list[int]]:
    """Longest-processing-time assignment of the main phase; cost cells run on shard 0 in the cost phase."""
    out: dict[str, list[int]] = {}
    load = [0.0] * nshards
    for c in sorted([c for c in cells if c.exp != "cost"], key=lambda c: (-c.cost(cfg), c.cell_id)):
        s = int(np.argmin(load))
        load[s] += c.cost(cfg)
        out[c.cell_id] = [s]
    for c in cells:
        if c.exp == "cost":
            out[c.cell_id] = [0]
    return out


def phase_of(cell: Cell) -> str:
    return "cost" if cell.exp == "cost" else "main"


# --------------------------------------------------------------------------- data


class DS:
    """One dataset with its pools, day splits and validity (shared by all cells of a process)."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.C = cfg["context_length"]
        self.H = cfg["horizon"]
        self.panel = load_panel(name)
        self.val = compute_validity(self.panel, self.C, self.H)
        pools = {k: np.asarray(v) for k, v in make_pools(self.panel, cfg["pools"], cfg["pool_seed"]).items()}
        rng = np.random.default_rng(stable_seed("targets", name, base=cfg["seed"]))
        vic = rng.permutation(pools["victim"])
        n_t = int(round(cfg["datasets"][name]["target_frac"] * len(vic)))
        pools["target"] = np.sort(vic[:n_t])
        pools["established"] = np.sort(vic[n_t:])
        pools["service"] = np.sort(np.concatenate([pools["established"], pools["aux"]]))
        pools["others_test"] = np.sort(np.concatenate([pools["victim"], pools["public"]]))
        pools["others_train"] = np.sort(np.concatenate([pools["aux"], pools["public"]]))
        pools["all"] = np.arange(self.panel.n_users)
        self.pools = pools
        self.valid = self.val.valid
        self.D = self.valid.shape[1]
        dcfg = cfg["datasets"][name]
        self.split = {
            "train": day_mask(self.val, dcfg.get("train_days"), dcfg.get("exclude_days")),
            "test": day_mask(self.val, dcfg.get("test_days"), dcfg.get("exclude_days")),
            "any": day_mask(self.val, None, dcfg.get("exclude_days")),
        }
        self.weekday = np.asarray(self.val.days.dayofweek)
        self._counts: dict[str, np.ndarray] = {}
        log.info("%s: %d users (victim %d = target %d + established %d, aux %d, public %d), %d origins",
                 name, self.panel.n_users, len(pools["victim"]), len(pools["target"]), len(pools["established"]),
                 len(pools["aux"]), len(pools["public"]), self.D)

    def count(self, pool: str) -> np.ndarray:
        if pool not in self._counts:
            self._counts[pool] = self.valid[self.pools[pool]].sum(axis=0)
        return self._counts[pool]

    def feasible(self, split: str, need: dict[str, int]) -> np.ndarray:
        ok = self.split[split].copy()
        for pool, n in need.items():
            ok &= self.count(pool) >= n
        return np.where(ok)[0]

    def users(self, pool: str, d: int) -> np.ndarray:
        u = self.pools[pool]
        return u[self.valid[u, d]]

    def ctx(self, u: int, d: int) -> np.ndarray:
        p = self.val.origins[d]
        return self.panel.values[u, p - self.C : p].astype(np.float32)

    def hor(self, u: int, d: int) -> np.ndarray:
        p = self.val.origins[d]
        return self.panel.values[u, p : p + self.H].astype(np.float32)


def pick(rng: np.random.Generator, ds: DS, pool: str, d: int, n: int, exclude) -> np.ndarray | None:
    cand = ds.users(pool, d)
    if exclude is not None and len(exclude):
        cand = cand[~np.isin(cand, np.asarray(exclude))]
    if len(cand) < n:
        return None
    return rng.choice(cand, size=n, replace=False) if n > 0 else np.zeros(0, dtype=int)


class Env:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._ds: dict[str, DS] = {}
        self._fc: dict[str, Forecaster] = {}
        self._centres: dict = {}
        self._releases: dict = {}
        self._refs: dict = {}

    def ds(self, name: str) -> DS:
        if name not in self._ds:
            self._ds[name] = DS(name, self.cfg)
        return self._ds[name]

    def refs(self, name: str) -> dict:
        if name not in self._refs:
            ds = self.ds(name)
            self._refs[name] = population_refs(ds, ds.pools["others_train"])
        return self._refs[name]

    def forecaster(self, tag: str, threads: int | None = None) -> Forecaster:
        key = (tag, threads)
        if key not in self._fc:
            self._fc[key] = Forecaster(self.cfg["models"][tag], threads=threads, horizon=self.cfg["horizon"])
        return self._fc[key]

    # ----------------------------------------------------------------- D4

    def centres(self, name: str, P: int) -> np.ndarray:
        """Public centres: k-means on public users' representations (training period for ELD, any day otherwise)."""
        key = (name, P)
        if key not in self._centres:
            ds = self.ds(name)
            kind = self.cfg["d4"]["representation"]
            rng = np.random.default_rng(stable_seed("centres", name, base=self.cfg["seed"]))
            split = "train" if self.cfg["datasets"][name].get("train_days") else "any"
            days = np.where(ds.split[split] & (ds.count("public") > 0))[0]
            days = rng.choice(days, size=min(len(days), int(self.cfg["d4"]["centre_days"])), replace=False)
            R = [dp.represent(ds.ctx(int(u), int(d)), kind) for d in days for u in ds.users("public", int(d))]
            self._centres[key] = dp.fit_centres(np.stack(R), P, stable_seed("kmeans", name, str(P), base=self.cfg["seed"]))
        return self._centres[key]

    def release(self, name: str, d: int, P: int, eps: float) -> dp.Release:
        """The day-d release from the service's established users (deterministic per day, shared by all cells)."""
        key = (name, d, P, eps)
        if key not in self._releases:
            ds = self.ds(name)
            kind = self.cfg["d4"]["representation"]
            users = ds.users("service", d)
            R = np.stack([dp.represent(ds.ctx(int(u), d), kind) for u in users]) if len(users) else np.zeros((0, dp.dim_of(kind, ds.C)))
            rng = np.random.default_rng(stable_seed("release", name, str(d), str(P), str(eps), base=self.cfg["seed"]))
            self._releases[key] = dp.release(R, self.centres(name, P), eps, float(self.cfg["delta"]), dp.default_clip(kind, ds.C), rng,
                                             float(self.cfg["d4"]["sum_share"]))
        return self._releases[key]


# --------------------------------------------------------------------------- helpers


def make_probe(kind: str, attacker_ctx: np.ndarray, rng: np.random.Generator, C: int, L: int) -> np.ndarray:
    if kind == "honest":
        return pad_left(attacker_ctx, C)
    if kind == "short":
        return pad_left(attacker_ctx[-L:], C)
    if kind == "noise":
        return pad_left(rng.standard_normal(L).astype(np.float32), C)
    raise ValueError(kind)


def filler_rows(kind: str, probe: np.ndarray, n: int, rng: np.random.Generator, C: int, L: int) -> list[np.ndarray]:
    """Sybil series submitted by the attacker besides its read-out probe."""
    if n <= 0:
        return []
    if kind == "copy":  # identical copies of the probe (look like ordinary new users)
        return [probe.copy() for _ in range(n)]
    if kind == "sparse":  # brand-new users with a single reading (the latest hour)
        one = np.full(C, np.nan, dtype=np.float32)
        one[-1] = 1.0
        return [one.copy() for _ in range(n)]
    if kind == "noise":  # independent 24-h noise series
        return [pad_left(rng.standard_normal(L).astype(np.float32), C) for _ in range(n)]
    raise ValueError(kind)


def eps_name(e) -> str:
    """Canonical name of a privacy level in configuration names: 0.5, 1, 2, 4, 8, inf."""
    e = float(e)
    return "inf" if math.isinf(e) else format(e, "g")


def d4_name(P: int, e) -> str:
    return f"d4_P{P}_eps{eps_name(e)}"


def n_attacker(fam: str, k: int) -> int:
    """Members the attacker controls (probe included)."""
    return {"single": 1, "sybil": k - 1, "half": k // 2}[fam]


def calendar_features(day) -> np.ndarray:
    f = np.zeros(19, dtype=np.float32)
    f[day.dayofweek] = 1.0
    f[7 + day.month - 1] = 1.0
    return f


def daily_means(ctx: np.ndarray) -> np.ndarray:
    """Daily means of the context in Chronos-2 instance-norm units (14 values, NaN if a day is missing)."""
    loc, scale = instance_loc_scale(ctx)
    z = (np.asarray(ctx, dtype=np.float64) - loc) / scale
    z = z.reshape(-1, 24)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(z, axis=1).astype(np.float32)


def day_evening_scores(ctx: np.ndarray) -> np.ndarray:
    """Evening score (18-23 h of the day's own z-profile) of each of the 14 days of a context (NaN if a day has < 2 values)."""
    days = np.asarray(ctx, dtype=np.float64).reshape(-1, 24)
    out = np.full(len(days), np.nan, dtype=np.float32)
    for i, d in enumerate(days):
        z = zprofile(d)
        if np.isfinite(z).all():
            out[i] = float(z[18:24].mean())
    return out


def self_residual(ev14: np.ndarray) -> float:
    """Last day's evening score minus the median of the previous 13 days (NaN if fewer than 7 previous days)."""
    prev = ev14[:-1]
    prev = prev[np.isfinite(prev)]
    if len(prev) < 7 or not np.isfinite(ev14[-1]):
        return float("nan")
    return float(ev14[-1] - np.median(prev))


def population_refs(ds: "DS", attacker_users: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per origin, over all users valid that day: mean previous-day z-profile, median evening score, median
    self-residual (see self_residual), number of users; and every (user, origin) previous-day z-profile.
    The same statistics over ``attacker_users`` only (keys with suffix ``_att``) define the attacker's training labels,
    so that no training label depends on victim-pool data.

    Computed from one z-profile per (user, calendar day); a context [p - 336, p) with p at midnight is exactly the
    14 calendar days before p, so the windows equal day_evening_scores() of the context."""
    vals = ds.panel.values
    U, D = ds.valid.shape
    if ds.panel.index[0].hour != 0:
        raise ValueError("panel must start at midnight")
    nd = vals.shape[1] // 24
    zday = np.full((U, nd, 24), np.nan, dtype=np.float32)
    for u in range(U):
        row = vals[u, : nd * 24].reshape(nd, 24)
        for t in range(nd):
            if np.isfinite(row[t]).sum() >= 2:
                zday[u, t] = zprofile(row[t])
    ev_day = zday[:, :, 18:24].mean(axis=2)
    tday = (np.asarray(ds.val.origins) // 24).astype(int)
    valid = ds.valid
    zp = np.where(valid[:, :, None], zday[:, tday - 1], np.nan).astype(np.float32)
    win = ev_day[:, tday[:, None] + np.arange(-14, 0)[None, :]]  # (U, D, 14)
    prev = win[..., :-1]
    cnt = np.isfinite(prev).sum(axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(prev, axis=2)
        res = np.where(valid & (cnt >= 7), win[..., -1] - med, np.nan)
        out = {"zp": zp, "n_valid": np.isfinite(zp[:, :, 0]).sum(axis=0)}
        sets = {"": np.arange(U)}
        if attacker_users is not None:
            sets["_att"] = np.asarray(attacker_users)
        for sfx, us in sets.items():
            out["ref_mean" + sfx] = np.nanmean(zp[us], axis=0).astype(np.float32)
            out["ref_even_med" + sfx] = np.nanmedian(zp[us][:, :, 18:24].mean(axis=2), axis=0).astype(np.float32)
            out["ref_resid_med" + sfx] = np.nanmedian(res[us], axis=0).astype(np.float32)
        return out


def _infeasible(reason: str) -> tuple[dict, dict]:
    return {"status": "infeasible", "reason": reason}, {}


def draw_day(rng: np.random.Generator, days: np.ndarray) -> int:
    return int(rng.choice(days))


# --------------------------------------------------------------------------- runners


def run_ref(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    """Population references per day and old-history candidate profiles (weekday-matched, 3..10 weeks back)."""
    ds = env.ds(cell.dataset)
    U, D = ds.valid.shape
    pr = env.refs(cell.dataset)
    zp = pr["zp"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stack = np.stack([np.roll(zp, 7 * j, axis=1) for j in range(3, 11)], axis=0)  # stack[j][u, d] = zp[u, d - 7j]
        for j_idx, j in enumerate(range(3, 11)):
            stack[j_idx][:, : 7 * j] = np.nan
        cnt = np.isfinite(stack[..., 0]).sum(axis=0)
        cand = np.nanmean(stack, axis=0)
    cand[cnt < 2] = np.nan
    n_valid = pr["n_valid"]
    summary = {"n_days": int(D), "n_users": int(U), "median_valid_per_day": float(np.median(n_valid)),
               "candidate_coverage": float(np.isfinite(cand[..., 0]).mean())}
    arr = {"ref_mean": pr["ref_mean"], "ref_even_med": pr["ref_even_med"], "ref_resid_med": pr["ref_resid_med"], "n_valid": n_valid,
           "ref_even_med_att": pr["ref_even_med_att"], "ref_resid_med_att": pr["ref_resid_med_att"],
           "cand": cand.astype(np.float16), "days": np.array([str(x.date()) for x in ds.val.days]),
           "weekday": ds.weekday, "pools_victim": ds.pools["victim"], "pools_aux": ds.pools["aux"], "pools_public": ds.pools["public"],
           "pools_target": ds.pools["target"]}
    return summary, arr


def run_util(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    ds = env.ds(cell.dataset)
    C, L = ds.C, p["ell"]
    xl, d1 = list(p["xl"]), list(p["d1"])
    kmax = max(xl) if xl else 1
    rng = np.random.default_rng(seed)
    days = ds.feasible("test", {"target": 1, "service": kmax - 1})
    if len(days) == 0:
        return _infeasible(f"no test origin with a target and {kmax - 1} established users")
    gap = int(cfg["d1"]["min_gap_days"])
    pub = ds.pools["public"]
    pub_valid = ds.valid[pub] & ds.split["any"][None, :]
    configs: list[str] = ["iso", "full"] + [f"xl{k}" for k in xl] + [f"d1_{k}" for k in d1]
    if p["d4"]:
        configs += [d4_name(P, e) for P in cfg["d4"]["P"] for e in cfg["d4"]["eps"]]
    groups, owner = [], []
    targets = []
    kind = cfg["d4"]["representation"]
    tries = 0
    while len(targets) < p["n"]:
        tries += 1
        if tries > 50 * p["n"]:
            break
        d = draw_day(rng, days)
        u = pick(rng, ds, "target", d, 1, None)
        cot = pick(rng, ds, "service", d, kmax - 1, None)
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
            # public windows: distinct public users, same weekday, at least `gap` days away from the target origin
            need = max(d1) - 1
            ok_days = np.where((ds.weekday == ds.weekday[d]) & (np.abs(np.arange(ds.D) - d) >= gap))[0]
            cand_users = [i for i in range(len(pub)) if pub_valid[i, ok_days].any()]
            if len(cand_users) < need:
                continue
            chosen = rng.choice(cand_users, size=need, replace=False)
            wins = []
            for i in chosen:
                dd = ok_days[pub_valid[i, ok_days]]
                wins.append(ds.ctx(int(pub[i]), int(rng.choice(dd))))
            for k in d1:
                rows[f"d1_{k}"] = [tgt] + wins[: k - 1]
        if p["d4"]:
            for P in cfg["d4"]["P"]:
                for e in cfg["d4"]["eps"]:
                    rel = env.release(cell.dataset, d, P, float(e))
                    rows[d4_name(P, e)] = [tgt] + [dp.to_series(r, kind, C) for r in rel.prototypes]
        for c in configs:
            groups.append(np.stack(rows[c]))
            owner.append((len(targets), c))
        targets.append((u, d, y, full))
    if not targets:
        return _infeasible("no target drawn")
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
               "wql_mean": {c: float(np.nanmean(W[c])) for c in configs},
               "gain_vs_iso": {c: float(1 - np.nanmean(W[c]) / np.nanmean(iso)) for c in configs}}
    return summary, arr


def _attack_roles(role: str) -> dict:
    if role == "train":
        return {"split": "train", "victim": "aux", "others": "others_train"}
    return {"split": "test", "victim": "victim", "others": "others_test"}


def run_attack(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    ds = env.ds(cell.dataset)
    C, L = ds.C, cfg["probe_short_len"]
    k, fam, filler, role = p["k"], p["fam"], p["filler"], p["role"]
    m = n_attacker(fam, k)
    n_oth = k - 1 - m
    R = _attack_roles(role)
    need = {"aux": 2 if R["victim"] == "aux" else 1}
    if R["victim"] != "aux":
        need[R["victim"]] = 1
    days = ds.feasible(R["split"], need)
    if n_oth > 0:
        days = days[ds.count(R["others"])[days] >= n_oth + 2]
    if len(days) == 0:
        return _infeasible(f"no {R['split']} origin for k={k} fam={fam}")
    rng = np.random.default_rng(seed)
    groups, probes, Y, dm, ev14, vic, att, dd, cal = [], [], [], [], [], [], [], [], []
    tries = 0
    while len(groups) < p["n"]:
        tries += 1
        if tries > 50 * p["n"]:
            break
        d = draw_day(rng, days)
        a = pick(rng, ds, "aux", d, 1, None)
        if a is None:
            continue
        a = int(a[0])
        v = pick(rng, ds, R["victim"], d, 1, [a])
        if v is None:
            continue
        v = int(v[0])
        oth = pick(rng, ds, R["others"], d, n_oth, [a, v])
        if oth is None:
            continue
        vctx = ds.ctx(v, d)
        z = zprofile(vctx[-24:])
        if not np.isfinite(z).all():
            continue
        probe = make_probe(p["probe"], ds.ctx(a, d), rng, C, L)
        rows = [probe, vctx] + filler_rows(filler, probe, m - 1, rng, C, L) + [ds.ctx(int(o), d) for o in oth]
        groups.append(np.stack(rows))
        probes.append(probe)
        Y.append(z)
        dm.append(daily_means(vctx))
        ev14.append(day_evening_scores(vctx))
        vic.append(v)
        att.append(a)
        dd.append(d)
        cal.append(calendar_features(ds.val.days[d]))
    if not groups:
        return _infeasible("no sample drawn")
    fc = env.forecaster(cell.model)
    Pxl = fc.predict_groups(groups)
    Piso = fc.predict_groups([pr[None] for pr in probes])
    fxl = np.stack([fc.model_space(Pxl[i][0], probes[i]).ravel() for i in range(len(probes))])
    fiso = np.stack([fc.model_space(Piso[i][0], probes[i]).ravel() for i in range(len(probes))])
    Y = np.stack(Y)
    summary = {"n": len(groups), "n_victims": int(len(np.unique(vic))), "n_attackers": int(len(np.unique(att))), "members_attacker": m,
               "members_honest_others": n_oth, "mean_abs_fxl_minus_fiso": float(np.abs(fxl - fiso).mean()),
               "evening_score_mean": float(evening_score(Y).mean())}
    arr = {"fxl": fxl.astype(np.float16), "fiso": fiso.astype(np.float16), "cal": np.stack(cal).astype(np.int8), "Y": Y.astype(np.float32),
           "dm": np.stack(dm).astype(np.float32), "ev14": np.stack(ev14).astype(np.float32), "victim": np.array(vic), "attacker": np.array(att),
           "day": np.array(dd)}
    return summary, arr


def run_channel(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    ds = env.ds(cell.dataset)
    C, L = ds.C, cfg["probe_short_len"]
    k, fam, filler = p["k"], p["fam"], p["filler"]
    m = n_attacker(fam, k)
    n_oth = k - 1 - m
    days = ds.feasible("test", {"aux": 1, "victim": 2})
    days = days[ds.count("all")[days] >= n_oth + 3]
    if len(days) == 0:
        return _infeasible(f"no test origin for k={k} fam={fam}")
    rng = np.random.default_rng(seed)
    gA, gB, probes, meta = [], [], [], []
    tries = 0
    while len(probes) < p["n"]:
        tries += 1
        if tries > 50 * p["n"]:
            break
        d = draw_day(rng, days)
        a = pick(rng, ds, "aux", d, 1, None)
        vv = pick(rng, ds, "victim", d, 2, None if a is None else [int(a[0])])
        if a is None or vv is None:
            continue
        a, v, v2 = int(a[0]), int(vv[0]), int(vv[1])
        oth = pick(rng, ds, "all", d, n_oth, [a, v, v2])
        if oth is None:
            continue
        probe = make_probe(p["probe"], ds.ctx(a, d), rng, C, L)
        fill = filler_rows(filler, probe, m - 1, rng, C, L)
        rest = [ds.ctx(int(o), d) for o in oth]
        gA.append(np.stack([probe, ds.ctx(v, d)] + fill + rest))
        gB.append(np.stack([probe, ds.ctx(v2, d)] + fill + rest))
        probes.append(probe)
        meta.append((a, v, v2, d))
    if not probes:
        return _infeasible("no pair drawn")
    fc = env.forecaster(cell.model)
    PA, PB = fc.predict_groups(gA), fc.predict_groups(gB)
    PI = fc.predict_groups([pr[None] for pr in probes])
    fA = np.stack([fc.model_space(PA[i][0], probes[i]) for i in range(len(probes))])
    fB = np.stack([fc.model_space(PB[i][0], probes[i]) for i in range(len(probes))])
    fI = np.stack([fc.model_space(PI[i][0], probes[i]) for i in range(len(probes))])
    i_swap = np.abs(fA - fB).mean(axis=(1, 2))
    i_group = np.abs(fA - fI).mean(axis=(1, 2))
    spread = (fA[:, 8, :] - fA[:, 0, :]).mean(axis=1)
    meta = np.array(meta)
    summary = {"n_pairs": len(probes), "i_swap_median": float(np.median(i_swap)), "i_swap_mean": float(np.mean(i_swap)),
               "i_group_median": float(np.median(i_group)), "spread_median": float(np.median(spread))}
    arr = {"i_swap": i_swap, "i_group": i_group, "spread": spread, "attacker": meta[:, 0], "victim": meta[:, 1], "victim_alt": meta[:, 2], "day": meta[:, 3]}
    return summary, arr


def run_invariance(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg = env.cfg
    ds = env.ds(cell.dataset)
    k = cell.p["k"]
    C, L = ds.C, cfg["probe_short_len"]
    days = ds.feasible("test", {"aux": 1, "victim": k - 1})
    if len(days) == 0:
        return _infeasible("no feasible origin")
    rng = np.random.default_rng(seed)
    fc = env.forecaster(cell.model)
    d_probe, d_equiv = [], []
    while len(d_probe) < 4 * cfg["invariance"]["n"]:
        d = draw_day(rng, days)
        a = pick(rng, ds, "aux", d, 1, None)
        vs = pick(rng, ds, "victim", d, k - 1, None)
        if a is None or vs is None:
            continue
        probe = make_probe("short", ds.ctx(int(a[0]), d), rng, C, L)
        v = ds.ctx(int(vs[0]), d)
        oth = [ds.ctx(int(o), d) for o in vs[1:]]
        s = float(np.nanstd(v)) + 1.0
        mu = float(abs(np.nanmean(v))) + 1.0
        transforms = [(2.0, 0.0), (0.5, 0.0), (1.0, 3.0 * s), (10.0, 5.0 * mu)]
        P = fc.predict_groups([np.stack([probe, v] + oth)] + [np.stack([probe, c * v + b] + oth) for c, b in transforms])
        f0 = fc.model_space(P[0][0], probe)
        scale_v = float(np.nanstd(v)) or 1.0
        for (c, b), Pt in zip(transforms, P[1:]):
            d_probe.append(float(np.abs(fc.model_space(Pt[0], probe) - f0).max()))
            d_equiv.append(float(np.abs(Pt[1] - (c * P[0][1] + b)).max() / (c * scale_v)))
    summary = {"n": cfg["invariance"]["n"], "probe_max_abs_change": float(np.max(d_probe)), "victim_equivariance_max_rel_err": float(np.max(d_equiv))}
    return summary, {"probe_change": np.array(d_probe), "victim_equiv_err": np.array(d_equiv)}


def run_dpaudit(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    """Adversary that knows every other user's data and sees the release: it observes the victim's clipped
    representation plus the Gaussian noise of the release, and trains a decoder on simulated releases of its own
    users (aux pool, training split) to infer the victim's day labels (victim pool, test split)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    from crossleak.metrics import auc

    cfg = env.cfg
    ds = env.ds(cell.dataset)
    kind = cfg["d4"]["representation"]
    Cn = dp.default_clip(kind, ds.C)
    from crossleak.attack import LABELS, labels

    rng = np.random.default_rng(seed)
    pr = env.refs(cell.dataset)

    def sample(role: str, n: int):
        R_ = _attack_roles(role)
        days = ds.feasible(R_["split"], {R_["victim"]: 1})
        X, Yp, DM, EV, D_, U_ = [], [], [], [], [], []
        while len(X) < n:
            d = draw_day(rng, days)
            u = int(pick(rng, ds, R_["victim"], d, 1, None)[0])
            c = ds.ctx(u, d)
            z = zprofile(c[-24:])
            if not np.isfinite(z).all():
                continue
            X.append(dp.clip_l2(dp.represent(c, kind)[None], Cn)[0])
            Yp.append(z)
            DM.append(daily_means(c))
            EV.append(day_evening_scores(c))
            D_.append(d)
            U_.append(u)
        return np.stack(X), np.stack(Yp), np.stack(DM), np.stack(EV), np.array(D_), np.array(U_)

    Xtr, Ytr, DMtr, EVtr, Dtr, Utr = sample("train", int(cfg["dpaudit"]["n_train"]))
    Xte, Yte, DMte, EVte, Dte, Ute = sample("test", int(cfg["dpaudit"]["n_test"]))
    thr = float(np.median(evening_score(Ytr)))
    lab_tr = labels(Ytr, DMtr, Dtr, pr["ref_even_med_att"], thr, EVtr, pr["ref_resid_med_att"])
    lab_te = labels(Yte, DMte, Dte, pr["ref_even_med"], thr, EVte, pr["ref_resid_med"])
    rows = []
    for P_eps in cfg["d4"]["eps"]:
        e = float(P_eps)
        sc = dp.noise_scales(e, float(cfg["delta"]), float(cfg["d4"]["sum_share"]))
        nz = sc.sigma_sum * Cn
        A = Xtr + (rng.normal(0, nz, Xtr.shape) if nz > 0 else 0)
        B = Xte + (rng.normal(0, nz, Xte.shape) if nz > 0 else 0)
        s = StandardScaler().fit(A)
        row = {"eps": e, "noise_std_per_coordinate": nz, "ceiling_attribute_auc": dp.attribute_ceiling(e, float(cfg["delta"]))}
        for lab in LABELS:
            ytr, yte = lab_tr[lab], lab_te[lab]
            if len(np.unique(ytr[ytr >= 0])) < 2:
                row[f"auc_{lab}"] = None
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegressionCV(Cs=8, cv=3, max_iter=2000, scoring="roc_auc").fit(s.transform(A)[ytr >= 0], ytr[ytr >= 0])
            row[f"auc_{lab}"] = auc(yte[yte >= 0], clf.decision_function(s.transform(B)[yte >= 0]))
        rows.append(row)
    summary = {"representation": kind, "clip": Cn, "n_train": len(Xtr), "n_test": len(Xte), "n_test_victims": int(len(np.unique(Ute))), "rows": rows}
    return summary, {"day_te": Dte, "victim_te": Ute}


def run_cost(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    """Latency and throughput per serving mode on this (idle) machine."""
    cfg = env.cfg
    cc = cfg["cost"]
    C = cfg["context_length"]
    import os

    threads = min(int(cc["threads"]), os.cpu_count() or 1)  # never oversubscribe: it distorts latency badly
    fc = env.forecaster(cell.model, threads=threads)
    rng = np.random.default_rng(seed)
    t = np.arange(C)

    def series(n: int) -> np.ndarray:
        return (np.sin(2 * np.pi * (t[None, :] + rng.integers(0, 24, (n, 1))) / 24) + 0.2 * rng.standard_normal((n, C))).astype(np.float32)

    def timed(groups: list[np.ndarray], repeats: int) -> np.ndarray:
        fc.predict_groups(groups[:1])
        out = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fc.predict_groups(groups)
            out.append(time.perf_counter() - t0)
        return np.array(out)

    R, B = int(cc["repeats"]), int(cc["batch_users"])
    rows = []

    def add(mode: str, k: int, users_per_call: int, groups: list[np.ndarray], repeats: int):
        ts = timed(groups, repeats)
        rows.append({"mode": mode, "group_size": k, "users_per_call": users_per_call, "series_per_call": int(sum(g.shape[0] for g in groups)),
                     "call_p50_s": float(np.median(ts)), "call_p95_s": float(np.quantile(ts, 0.95)),
                     "per_user_p50_s": float(np.median(ts) / users_per_call), "per_user_p95_s": float(np.quantile(ts, 0.95) / users_per_call)})

    add("iso_request", 1, 1, [series(1)], R)
    add("iso_batch", 1, B, [series(1) for _ in range(B)], max(3, R // 4))
    for k in cc["xl_k"]:
        add("xl_group", k, k, [series(k)], max(3, R // 2) if k >= 64 else R)
    kind = cfg["d4"]["representation"]
    for P in cc["d4_P"]:
        protos = [dp.to_series(rng.standard_normal(dp.dim_of(kind, C)), kind, C) for _ in range(P)]
        add("d4_request", P + 1, 1, [np.stack([series(1)[0]] + protos)], R)
        add("d4_batch", P + 1, B, [np.stack([series(1)[0]] + protos) for _ in range(B)], max(3, R // 4))
    # release cost for 1,000 users (represent + clip + assign + noise), P = largest
    P = max(cc["d4_P"])
    raw = series(1000)
    cent = dp.fit_centres(np.stack([dp.represent(x, kind) for x in raw[:200]]), P, seed)
    t0 = time.perf_counter()
    Rr = np.stack([dp.represent(x, kind) for x in raw])
    dp.release(Rr, cent, 1.0, float(cfg["delta"]), dp.default_clip(kind, C), rng)
    rel_s = time.perf_counter() - t0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {"threads": threads, "rows": rows, "release_1000_users_s": rel_s, "peak_rss_mb": rss_mb}
    return summary, {}


RUNNERS = {"ref": run_ref, "util": run_util, "attack": run_attack, "channel": run_channel, "invariance": run_invariance,
           "dpaudit": run_dpaudit, "cost": run_cost}


# --------------------------------------------------------------------------- driver


def cell_dir(cfg: dict, cell: Cell) -> Path:
    return runs_dir() / cfg["experiment"] / cell.cell_id


def cell_status(cfg: dict, cell: Cell) -> str:
    d = cell_dir(cfg, cell)
    if (d / "DONE").exists():
        return "done"
    if (d / "FAILED").exists():
        return "failed"
    return "todo"


def failed_attempts(cfg: dict, cell: Cell) -> int:
    f = cell_dir(cfg, cell) / "FAILED"
    if not f.exists():
        return 0
    try:
        return int(json.loads(f.read_text()).get("attempts", 1))
    except Exception:  # noqa: BLE001
        return 1


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and math.isinf(o):
        return "inf"
    return str(o)


def _short(cell: Cell, s: dict) -> str:
    if s.get("status") == "infeasible":
        return f"infeasible: {s.get('reason')}"
    if cell.exp == "util":
        g = s["gain_vs_iso"]
        keys = [c for c in g if c.startswith("xl")][:2] + [c for c in g if c.startswith("d4") and "inf" not in c][:1]
        return f"n={s['n_targets']} " + " ".join(f"{c}={g[c]:+.3f}" for c in keys)
    if cell.exp == "attack":
        return f"n={s['n']} victims={s['n_victims']} |fxl-fiso|={s['mean_abs_fxl_minus_fiso']:.4f}"
    if cell.exp == "channel":
        return f"n={s['n_pairs']} i_swap_med={s['i_swap_median']:.4f}"
    if cell.exp == "invariance":
        return f"probe_max_change={s['probe_max_abs_change']:.2e}"
    if cell.exp == "dpaudit":
        return " ".join(f"eps={r['eps']}:{(r.get('auc_L1') or float('nan')):.3f}" for r in s["rows"])
    if cell.exp == "cost":
        return "; ".join(f"{r['mode']}{r['group_size']}:{r['per_user_p50_s']*1000:.0f}ms/user" for r in s["rows"])
    if cell.exp == "ref":
        return f"median valid/day {s['median_valid_per_day']:.0f}"
    return ""


def run_one(cell: Cell, env: Env) -> str:
    cfg = env.cfg
    d = cell_dir(cfg, cell)
    d.mkdir(parents=True, exist_ok=True)
    seed = stable_seed(cell.sample_key, base=cfg["seed"])
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    attempts = failed_attempts(cfg, cell) + 1
    log.info("cell start %s (seed %d, attempt %d, est %.0fs)", cell.cell_id, seed, attempts, cell.cost(cfg))
    try:
        summary, arr = RUNNERS[cell.exp](cell, env, seed)
        result = {"cell_id": cell.cell_id, "family_key": cell.family_key, "exp": cell.exp, "dataset": cell.dataset, "model": cell.model,
                  "model_id": cfg["models"].get(cell.model), "params": cell.p, "seed": seed, "version": TAG, "host": hostname(),
                  "started": started, "duration_sec": round(time.time() - t0, 1), "config_sha256": cfg["_sha256"],
                  "status": summary.get("status", "ok"), "summary": summary}
        (d / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
        if arr:
            np.savez_compressed(d / "arrays.npz", **arr)
        (d / "DONE").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": hostname(), "version": TAG}))
        if (d / "FAILED").exists():
            (d / "FAILED").unlink()
        log.info("cell done %s in %.1fs: %s", cell.cell_id, time.time() - t0, _short(cell, summary))
        return "done"
    except Exception as exc:  # noqa: BLE001 - one failing cell must not stop the campaign
        tb = traceback.format_exc()
        (d / "FAILED").write_text(json.dumps({"attempts": attempts, "error": repr(exc), "traceback": tb, "host": hostname(), "version": TAG}, indent=2))
        log.error("cell FAILED %s: %r\n%s", cell.cell_id, exc, tb)
        return "failed"


def shard_cells(cfg: dict, shard: int, nshards: int, phase: str = "main", only: list[str] | None = None) -> list[Cell]:
    cells = build_cells(cfg)
    assign = assign_shards(cells, cfg, nshards)
    mine = [c for c in cells if shard in assign[c.cell_id] and phase_of(c) == phase and (not only or c.exp in only)]
    mine.sort(key=lambda c: (c.priority, -c.cost(cfg), c.cell_id))
    return mine


def run_campaign(cfg: dict, shard: int, nshards: int, budget_min: float | None, only: list[str] | None, phase: str = "main") -> dict:
    mine = shard_cells(cfg, shard, nshards, phase, only)
    plan_path = runs_dir() / cfg["experiment"] / f"plan_{phase}_shard{shard}of{nshards}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"version": TAG, "host": hostname(), "phase": phase, "shard": shard, "nshards": nshards,
                                     "config_sha256": cfg["_sha256"], "cells": [c.cell_id for c in mine],
                                     "est_sec": {c.cell_id: round(c.cost(cfg)) for c in mine}}, indent=2))
    env = Env(cfg)
    t0 = time.time()
    counts = {"done": 0, "failed": 0, "skipped": 0, "not_started": 0}
    log.info("campaign %s phase %s: shard %d/%d, %d cells, estimated %.1f h", cfg["experiment"], phase, shard, nshards, len(mine),
             sum(c.cost(cfg) for c in mine) / 3600)
    for i, c in enumerate(mine):
        st = cell_status(cfg, c)
        if st == "done":
            counts["skipped"] += 1
            continue
        if st == "failed" and failed_attempts(cfg, c) >= 2:
            log.warning("skip %s: failed twice already", c.cell_id)
            counts["failed"] += 1
            continue
        if budget_min is not None and (time.time() - t0) / 60.0 > budget_min:
            counts["not_started"] = len(mine) - i
            log.warning("time budget of %.0f min reached; %d cells not started (rerun the same command to resume)", budget_min, counts["not_started"])
            break
        counts[run_one(c, env)] += 1
        log.info("progress %d/%d (elapsed %.1f min)", i + 1, len(mine), (time.time() - t0) / 60.0)
    log.info("campaign finished: %s", counts)
    return counts
