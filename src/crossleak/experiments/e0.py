"""E0: pilot and killer experiment (gain, channel, decodability, invariance, speed).

Every cell is independent, resumable and writes:
  runs/<exp_name>/<cell_id>/result.json   summary numbers
  runs/<exp_name>/<cell_id>/arrays.npz    per-sample data for independent re-analysis
  runs/<exp_name>/<cell_id>/DONE | FAILED  status marker
"""

from __future__ import annotations

import json
import logging
import math
import resource
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crossleak.config import TAG, hostname, runs_dir, stable_seed
from crossleak.data.panel import Sampler, compute_validity, day_mask, load_panel, make_pools
from crossleak.decode import decode_eval, evening_score, zprofile
from crossleak.metrics import cluster_bootstrap, mase_item, wql_item, wql_point
from crossleak.model import Forecaster, pad_left

log = logging.getLogger("crossleak.e0")

MODEL_COST = {"c2small": 1.0, "c2base": 4.4}
EXP_ORDER = {"speed": 0, "invariance": 1, "decode": 2, "channel": 3, "gain": 4}


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
        """Identifies the sampled groups; independent of the model so models are compared on identical groups."""
        return "__".join([self.exp, self.dataset] + [f"{k}{v}" for k, v in self.params])

    @property
    def cell_id(self) -> str:
        return "__".join([self.exp, self.dataset, self.model] + [f"{k}{v}" for k, v in self.params])

    def cost(self, cfg: dict) -> float:
        p, m = self.p, MODEL_COST.get(self.model, 1.0)
        if self.exp == "gain":
            n = cfg["gain"]["n_targets"]
            per = (p["k"] + 2) if p["scen"] == "mixed" else 2
            return n * per * m
        if self.exp == "channel":
            return cfg["channel"]["n_pairs"] * (2 * p["k"] + 1) * m
        if self.exp == "decode":
            return (cfg["decode"]["n_train"] + cfg["decode"]["n_test"]) * (p["k"] + 1) * m
        if self.exp == "invariance":
            return cfg["invariance"]["n"] * 5 * cfg["invariance"]["k"] * m
        return 0.0


def build_cells(cfg: dict) -> list[Cell]:
    cells: list[Cell] = []
    for m in cfg["models"]:
        cells.append(Cell("speed", "synthetic", m, ()))
        for ds, dcfg in cfg["datasets"].items():
            cells.append(Cell("invariance", ds, m, (("k", cfg["invariance"]["k"]),)))
            for k in dcfg["decode_k"]:
                for probe in cfg["decode"]["probes"]:
                    cells.append(Cell("decode", ds, m, (("k", k), ("probe", probe))))
            for k in dcfg["channel_k"]:
                for probe in cfg["channel"]["probes"]:
                    cells.append(Cell("channel", ds, m, (("k", k), ("probe", probe))))
            for scen in cfg["gain"]["scenarios"]:
                for ell in cfg["gain"]["target_context"]:
                    for k in dcfg["gain_k"]:
                        cells.append(Cell("gain", ds, m, (("ell", ell), ("k", k), ("scen", scen))))
    return cells


def assign_shards(cells: list[Cell], cfg: dict, nshards: int) -> dict[str, list[int]]:
    """Longest-processing-time assignment; speed cells run on every shard (one per machine)."""
    out: dict[str, list[int]] = {}
    load = [0.0] * nshards
    work = [c for c in cells if c.exp != "speed"]
    for c in sorted(work, key=lambda c: (-c.cost(cfg), c.cell_id)):
        s = int(np.argmin(load))
        load[s] += c.cost(cfg)
        out[c.cell_id] = [s]
    for c in cells:
        if c.exp == "speed":
            out[c.cell_id] = list(range(nshards))
    return out


class Env:
    """Lazily loaded panels, validity, pools and forecasters."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._panels: dict = {}
        self._fc: dict = {}

    def forecaster(self, tag: str) -> Forecaster:
        if tag not in self._fc:
            self._fc[tag] = Forecaster(self.cfg["models"][tag], horizon=self.cfg["horizon"])
        return self._fc[tag]

    def data(self, ds: str):
        if ds not in self._panels:
            panel = load_panel(ds)
            val = compute_validity(panel, self.cfg["context_length"], self.cfg["horizon"])
            pools = make_pools(panel, self.cfg["pools"], self.cfg["seed"])
            self._panels[ds] = (panel, val, pools)
            log.info(
                "%s: %d users (victim %d, aux %d, public %d), %d origins, %.1f%% valid (user, origin) pairs",
                ds, panel.n_users, len(pools["victim"]), len(pools["aux"]), len(pools["public"]),
                len(val.origins), 100 * val.valid.mean(),
            )
        return self._panels[ds]

    def sampler(self, ds: str, split: str, seed: int) -> Sampler:
        panel, val, pools = self.data(ds)
        dcfg = self.cfg["datasets"][ds]
        ok = day_mask(val, dcfg.get(f"{split}_days"), dcfg.get("exclude_days"))
        return Sampler(panel, val, pools, ok, np.random.default_rng(seed))


# --------------------------------------------------------------------------- helpers


def make_probe(kind: str, attacker_ctx: np.ndarray, rng: np.random.Generator, C: int, short_len: int) -> np.ndarray:
    if kind == "honest":
        return pad_left(attacker_ctx, C)
    if kind == "short":
        return pad_left(attacker_ctx[-short_len:], C)
    if kind == "noise":
        return pad_left(rng.standard_normal(short_len).astype(np.float32), C)
    raise ValueError(kind)


def calendar_features(day) -> np.ndarray:
    f = np.zeros(19, dtype=np.float32)
    f[day.dayofweek] = 1.0
    f[7 + day.month - 1] = 1.0
    return f


def _infeasible(cell: Cell, reason: str) -> dict:
    return {"status": "infeasible", "reason": reason}


# --------------------------------------------------------------------------- runners


def run_gain(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    k, ell, scen = p["k"], p["ell"], p["scen"]
    C = cfg["context_length"]
    S = env.sampler(cell.dataset, "test", seed)
    need = {"victim": k}
    days = S.feasible_days(need)
    if len(days) == 0:
        return _infeasible(cell, f"no origin with {k} valid victims"), {}
    n_targets = cfg["gain"]["n_targets"]
    n_groups = n_targets if scen == "mixed" else math.ceil(n_targets / k)
    groups, iso, full, targets = [], [], [], []
    for _ in range(n_groups):
        d, pk = S.draw(need, days)
        us = pk["victim"]
        ctx = {int(u): S.context(int(u), d) for u in us}
        if scen == "mixed":
            t = int(us[0])
            rows = [pad_left(ctx[t][-ell:], C)] + [ctx[int(u)] for u in us[1:]]
            groups.append(np.stack(rows))
            iso.append(ctx[t][-ell:][None])
            full.append(ctx[t][None])
            targets.append((t, d, 0))
        else:
            groups.append(np.stack([pad_left(ctx[int(u)][-ell:], C) for u in us]))
            for j, u in enumerate(us):
                iso.append(ctx[int(u)][-ell:][None])
                targets.append((int(u), d, j))
    fc = env.forecaster(cell.model)
    P_xl = fc.predict_groups(groups)
    P_iso = fc.predict_groups(iso)
    P_full = fc.predict_groups(full) if full else []
    rows = []
    gi = 0
    for i, (u, d, j) in enumerate(targets):
        if scen == "mixed":
            gi = i
        else:
            gi = i // k
        y = S.horizon(u, d)
        hist = S.context(u, d)
        q_xl, q_iso = P_xl[gi][j], P_iso[i][0]
        rec = {
            "user": u,
            "day": d,
            "wql_xl": wql_item(y, q_xl),
            "wql_iso": wql_item(y, q_iso),
            "wql_snaive": wql_point(y, hist[-24:]),
            "mase_xl": mase_item(y, q_xl[4], hist),
            "mase_iso": mase_item(y, q_iso[4], hist),
            "wql_full": wql_item(y, P_full[i][0]) if P_full else np.nan,
        }
        rows.append(rec)
    arr = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    ok = np.isfinite(arr["wql_xl"]) & np.isfinite(arr["wql_iso"])
    wx, wi, users = arr["wql_xl"][ok], arr["wql_iso"][ok], arr["user"][ok]
    boot = cluster_bootstrap(lambda i: 1.0 - wx[i].mean() / wi[i].mean(), users, cfg["bootstrap"], seed)
    rel = 1.0 - wx / wi
    summary = {
        "n_targets": int(ok.sum()),
        "n_users": int(len(np.unique(users))),
        "wql_xl_mean": float(wx.mean()),
        "wql_iso_mean": float(wi.mean()),
        "wql_full_mean": float(np.nanmean(arr["wql_full"][ok])) if scen == "mixed" else None,
        "wql_snaive_mean": float(np.nanmean(arr["wql_snaive"][ok])),
        "gain": boot,
        "gain_median_item": float(np.median(rel)),
        "share_items_improved": float((rel > 0).mean()),
        "mase_xl_mean": float(np.nanmean(arr["mase_xl"][ok])),
        "mase_iso_mean": float(np.nanmean(arr["mase_iso"][ok])),
    }
    return summary, arr


def run_channel(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    k, probe_kind = p["k"], p["probe"]
    C, L = cfg["context_length"], cfg["probe_short_len"]
    S = env.sampler(cell.dataset, "test", seed)
    need = {"aux": 1, "victim": k}
    days = S.feasible_days(need)
    if len(days) == 0:
        return _infeasible(cell, f"no origin with 1 aux and {k} victims"), {}
    rng = np.random.default_rng(seed + 7)
    gA, gB, probes, meta = [], [], [], []
    for _ in range(cfg["channel"]["n_pairs"]):
        d, pk = S.draw(need, days)
        a = int(pk["aux"][0])
        v, v2, others = int(pk["victim"][0]), int(pk["victim"][1]), [int(x) for x in pk["victim"][2:]]
        probe = make_probe(probe_kind, S.context(a, d), rng, C, L)
        oth = [S.context(o, d) for o in others]
        gA.append(np.stack([probe, S.context(v, d)] + oth))
        gB.append(np.stack([probe, S.context(v2, d)] + oth))
        probes.append(probe)
        meta.append((a, v, v2, d))
    fc = env.forecaster(cell.model)
    PA, PB = fc.predict_groups(gA), fc.predict_groups(gB)
    PI = fc.predict_groups([pr[None] for pr in probes])
    fA = np.stack([fc.model_space(PA[i][0], probes[i]) for i in range(len(probes))])
    fB = np.stack([fc.model_space(PB[i][0], probes[i]) for i in range(len(probes))])
    fI = np.stack([fc.model_space(PI[i][0], probes[i]) for i in range(len(probes))])
    i_swap = np.abs(fA - fB).mean(axis=(1, 2))
    i_group = np.abs(fA - fI).mean(axis=(1, 2))
    spread = (fA[:, 8, :] - fA[:, 0, :]).mean(axis=1)
    summary = {
        "n_pairs": len(probes),
        "i_swap_median": float(np.median(i_swap)),
        "i_swap_q90": float(np.quantile(i_swap, 0.9)),
        "i_group_median": float(np.median(i_group)),
        "spread_median": float(np.median(spread)),
        "swap_to_spread_median": float(np.median(i_swap / np.maximum(spread, 1e-6))),
    }
    meta = np.array(meta)
    arr = {"i_swap": i_swap, "i_group": i_group, "spread": spread, "attacker": meta[:, 0], "victim": meta[:, 1], "victim_alt": meta[:, 2], "day": meta[:, 3]}
    return summary, arr


def _decode_samples(S: Sampler, need: dict, roles: str, n: int, k: int, probe_kind: str, C: int, L: int, rng) -> dict:
    days = S.feasible_days(need)
    if len(days) == 0:
        raise RuntimeError(f"no feasible origin for {need}")
    groups, probes, Y, att, vic, dd, cal = [], [], [], [], [], [], []
    while len(groups) < n:
        d, pk = S.draw(need, days)
        if roles == "train":
            us = [int(x) for x in pk["aux"]]
            a, v, others = us[0], us[1], us[2:]
        else:
            a = int(pk["aux"][0])
            vs = [int(x) for x in pk["victim"]]
            v, others = vs[0], vs[1:]
        vctx = S.context(v, d)
        z = zprofile(vctx[-24:])
        if not np.isfinite(z).all():
            continue
        probe = make_probe(probe_kind, S.context(a, d), rng, C, L)
        groups.append(np.stack([probe, vctx] + [S.context(o, d) for o in others]))
        probes.append(probe)
        Y.append(z)
        att.append(a)
        vic.append(v)
        dd.append(d)
        cal.append(calendar_features(S.validity.days[d]))
    return {"groups": groups, "probes": probes, "Y": np.stack(Y), "attacker": np.array(att), "victim": np.array(vic), "day": np.array(dd), "cal": np.stack(cal)}


def run_decode(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    k, probe_kind = p["k"], p["probe"]
    C, L = cfg["context_length"], cfg["probe_short_len"]
    rng = np.random.default_rng(seed + 11)
    S_tr = env.sampler(cell.dataset, "train", seed)
    S_te = env.sampler(cell.dataset, "test", seed + 1)
    try:
        tr = _decode_samples(S_tr, {"aux": k}, "train", cfg["decode"]["n_train"], k, probe_kind, C, L, rng)
        te = _decode_samples(S_te, {"aux": 1, "victim": k - 1}, "test", cfg["decode"]["n_test"], k, probe_kind, C, L, rng)
    except RuntimeError as exc:
        return _infeasible(cell, str(exc)), {}
    fc = env.forecaster(cell.model)

    def feats(part: dict) -> tuple[np.ndarray, np.ndarray]:
        Pxl = fc.predict_groups(part["groups"])
        Piso = fc.predict_groups([pr[None] for pr in part["probes"]])
        fxl = np.stack([fc.model_space(Pxl[i][0], part["probes"][i]).ravel() for i in range(len(Pxl))])
        fiso = np.stack([fc.model_space(Piso[i][0], part["probes"][i]).ravel() for i in range(len(Piso))])
        return fxl, fiso

    fxl_tr, fiso_tr = feats(tr)
    fxl_te, fiso_te = feats(te)
    Xtr_xl = np.hstack([fxl_tr, fiso_tr, tr["cal"]])
    Xtr_c = np.hstack([fiso_tr, tr["cal"]])
    Xte_xl = np.hstack([fxl_te, fiso_te, te["cal"]])
    Xte_c = np.hstack([fiso_te, te["cal"]])
    s_tr, s_te = evening_score(tr["Y"]), evening_score(te["Y"])
    res = decode_eval(Xtr_xl, Xtr_c, tr["Y"], s_tr, Xte_xl, Xte_c, te["Y"], s_te, te["victim"], seed, cfg["bootstrap"])
    res["n_train"], res["n_test"] = len(tr["Y"]), len(te["Y"])
    res["n_test_victims"] = int(len(np.unique(te["victim"])))
    res["n_train_users"] = int(len(np.unique(np.concatenate([tr["attacker"], tr["victim"]]))))
    arr = {
        "fxl_tr": fxl_tr.astype(np.float16), "fiso_tr": fiso_tr.astype(np.float16), "cal_tr": tr["cal"].astype(np.int8),
        "Y_tr": tr["Y"].astype(np.float32), "victim_tr": tr["victim"], "attacker_tr": tr["attacker"], "day_tr": tr["day"],
        "fxl_te": fxl_te.astype(np.float16), "fiso_te": fiso_te.astype(np.float16), "cal_te": te["cal"].astype(np.int8),
        "Y_te": te["Y"].astype(np.float32), "victim_te": te["victim"], "attacker_te": te["attacker"], "day_te": te["day"],
    }
    return res, arr


def run_invariance(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg = env.cfg
    k = cell.p["k"]
    C, L = cfg["context_length"], cfg["probe_short_len"]
    S = env.sampler(cell.dataset, "test", seed)
    need = {"aux": 1, "victim": k - 1}
    days = S.feasible_days(need)
    if len(days) == 0:
        return _infeasible(cell, "no feasible origin"), {}
    rng = np.random.default_rng(seed + 3)
    fc = env.forecaster(cell.model)
    d_probe, d_equiv = [], []
    for _ in range(cfg["invariance"]["n"]):
        d, pk = S.draw(need, days)
        a = int(pk["aux"][0])
        vs = [int(x) for x in pk["victim"]]
        probe = make_probe("short", S.context(a, d), rng, C, L)
        v = S.context(vs[0], d)
        oth = [S.context(o, d) for o in vs[1:]]
        s = float(np.nanstd(v)) + 1.0
        mu = float(abs(np.nanmean(v))) + 1.0
        transforms = [(2.0, 0.0), (0.5, 0.0), (1.0, 3.0 * s), (10.0, 5.0 * mu)]
        groups = [np.stack([probe, v] + oth)] + [np.stack([probe, c * v + b] + oth) for c, b in transforms]
        P = fc.predict_groups(groups)
        f0 = fc.model_space(P[0][0], probe)
        _, scale_v = (float(np.nanmean(v)), float(np.nanstd(v)) or 1.0)
        for (c, b), Pt in zip(transforms, P[1:]):
            d_probe.append(float(np.abs(fc.model_space(Pt[0], probe) - f0).max()))
            d_equiv.append(float(np.abs(Pt[1] - (c * P[0][1] + b)).max() / (c * scale_v)))
    summary = {"n": cfg["invariance"]["n"], "probe_max_abs_change": float(np.max(d_probe)), "victim_equivariance_max_rel_err": float(np.max(d_equiv))}
    return summary, {"probe_change": np.array(d_probe), "victim_equiv_err": np.array(d_equiv)}


def run_speed(cell: Cell, env: Env, seed: int) -> tuple[dict, dict]:
    cfg = env.cfg
    C = cfg["context_length"]
    fc = env.forecaster(cell.model)
    rng = np.random.default_rng(seed)
    t = np.arange(C)
    rows = []
    for k in cfg["speed"]["k"]:
        n_tasks = max(1, 256 // k)
        groups = [(np.sin(2 * np.pi * (t[None, :] + rng.integers(0, 24, (k, 1))) / 24) + 0.2 * rng.standard_normal((k, C))).astype(np.float32) for _ in range(n_tasks)]
        fc.predict_groups(groups[:1])
        times = []
        for _ in range(cfg["speed"]["repeats"]):
            t0 = time.perf_counter()
            fc.predict_groups(groups)
            times.append(time.perf_counter() - t0)
        per_series = np.array(times) / (n_tasks * k)
        rows.append({"k": k, "n_tasks": n_tasks, "sec_per_series_p50": float(np.median(per_series)), "sec_per_series_max": float(per_series.max()), "sec_per_group_p50": float(np.median(times) / n_tasks)})
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    summary = {"rows": rows, "threads": fc.threads, "peak_rss_mb": rss_mb}
    return summary, {}


RUNNERS = {"gain": run_gain, "channel": run_channel, "decode": run_decode, "invariance": run_invariance, "speed": run_speed}


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


def run_one(cell: Cell, env: Env) -> str:
    cfg = env.cfg
    d = cell_dir(cfg, cell)
    d.mkdir(parents=True, exist_ok=True)
    seed = stable_seed(cell.sample_key, base=cfg["seed"])
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    attempts = failed_attempts(cfg, cell) + 1
    log.info("cell start %s (seed %d, attempt %d)", cell.cell_id, seed, attempts)
    try:
        summary, arr = RUNNERS[cell.exp](cell, env, seed)
        result = {
            "cell_id": cell.cell_id, "exp": cell.exp, "dataset": cell.dataset, "model": cell.model,
            "model_id": cfg["models"][cell.model], "params": cell.p, "seed": seed, "version": TAG,
            "host": hostname(), "started": started, "duration_sec": round(time.time() - t0, 1),
            "config_sha256": cfg["_sha256"], "status": summary.get("status", "ok"), "summary": summary,
        }
        (d / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
        if arr:
            np.savez_compressed(d / "arrays.npz", **arr)
        (d / "DONE").write_text(json.dumps({"finished": time.strftime("%Y-%m-%dT%H:%M:%S"), "host": hostname(), "version": TAG}))
        if (d / "FAILED").exists():
            (d / "FAILED").unlink()
        log.info("cell done %s in %.1fs: %s", cell.cell_id, time.time() - t0, _short(result))
        return "done"
    except Exception as exc:  # noqa: BLE001 - a failing cell must not stop the campaign
        tb = traceback.format_exc()
        (d / "FAILED").write_text(json.dumps({"attempts": attempts, "error": repr(exc), "traceback": tb, "host": hostname(), "version": TAG}, indent=2))
        log.error("cell FAILED %s: %r\n%s", cell.cell_id, exc, tb)
        return "failed"


def _short(result: dict) -> str:
    s = result["summary"]
    if result["exp"] == "gain" and "gain" in s:
        g = s["gain"]
        return f"gain={g['est']:.4f} [{g['lo']:.4f},{g['hi']:.4f}] n={s['n_targets']}"
    if result["exp"] == "decode" and "delta_r" in s:
        return f"delta_r={s['delta_r']['est']:.3f} [{s['delta_r']['lo']:.3f},{s['delta_r']['hi']:.3f}] auc_xl={s['auc_xl']['est']:.3f} auc_ctrl={s['auc_ctrl']:.3f}"
    if result["exp"] == "channel" and "i_swap_median" in s:
        return f"i_swap_med={s['i_swap_median']:.4f} swap/spread={s['swap_to_spread_median']:.3f}"
    if result["exp"] == "invariance" and "probe_max_abs_change" in s:
        return f"probe_max_change={s['probe_max_abs_change']:.2e} equiv_err={s['victim_equivariance_max_rel_err']:.2e}"
    if result["exp"] == "speed":
        return "; ".join(f"k={r['k']}:{r['sec_per_series_p50']*1000:.1f}ms/series" for r in s["rows"])
    return s.get("status", "")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run_campaign(cfg: dict, shard: int, nshards: int, budget_min: float | None, only: list[str] | None) -> dict:
    cells = build_cells(cfg)
    assign = assign_shards(cells, cfg, nshards)
    mine = [c for c in cells if shard in assign[c.cell_id] and (not only or c.exp in only)]
    mine.sort(key=lambda c: (EXP_ORDER[c.exp], c.dataset, c.model, c.cell_id))
    plan_path = runs_dir() / cfg["experiment"] / f"plan_shard{shard}of{nshards}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps({"version": TAG, "host": hostname(), "shard": shard, "nshards": nshards, "config_sha256": cfg["_sha256"], "cells": [c.cell_id for c in mine]}, indent=2))
    env = Env(cfg)
    t0 = time.time()
    counts = {"done": 0, "failed": 0, "skipped": 0, "not_started": 0}
    log.info("campaign %s: shard %d/%d, %d cells assigned", cfg["experiment"], shard, nshards, len(mine))
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
