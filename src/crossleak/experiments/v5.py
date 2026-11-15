"""v0.5 study: evidence requested in the manuscript review, round r3 (prereg/V5_PREREG.md).

Reuses the frozen main driver (``experiments/main.py``, source unchanged) for its data, cells, attack runner and
campaign machinery, and adds:

  E5a  util       the public-context ablation on the same cold-start targets (base model, one day of history):
                    d1p_<k>  k - 1 public windows of the same weekday, >= min_gap_days before the origin (as v0.4)
                    d1x_<k>  the same, but on a *different* weekday
                    d1r_<k>  the same, on any weekday (random historical windows)
                    d1o_<k>  windows of public users of the *other* dataset (other domain), any day
                  next to iso, full and xl_<k>.
  E5c  attack     the stronger sybil attacker of v0.3 at k = 256 and k = 32, repeated over independent realisations
                  (``rep`` is part of the sample key, so every realisation draws its own training and test groups).
  E5d  attack     half sybils (128 of 256 members) at k = 256;
       placement  a service-level emulation of batch placement (no model): co-location probability, attacker
                  occupancy and attacker cost under explicit grouping policies.
  E5b  cost       an end-to-end nightly serving run for 1,000 new clients on one machine (phase "cost"): D0, XL, D1
                  (past-only), D4, seasonal naive and a ridge baseline; total time, per-call p50/p95, peak RSS, throughput.

:func:`install` points the main module's ``build_cells``, its utility, cost and placement runners and its log-line
formatter at this module (the environment class is the main one).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from crossleak.config import stable_seed
from crossleak.defenses import dp_prototype as dp
from crossleak.experiments import main as M
from crossleak.experiments import v4 as _v4
from crossleak.metrics import wql_item
from crossleak.model import pad_left

log = logging.getLogger("crossleak.v5")

# pristine main-driver objects: v4 snapshots them when it is first imported (before any patch), the rest v4 never patches
_ORIGINAL = {"build_cells": _v4._ORIGINAL["build_cells"], "util": _v4._ORIGINAL["util"], "cost": M.RUNNERS["cost"],
             "placement": M.RUNNERS.get("placement"), "short": M._short}

ARMS = ("d1p", "d1x", "d1r", "d1o")


# --------------------------------------------------------------------------- cells


@dataclass(frozen=True)
class V5Cell(M.Cell):
    """Main cell with v0.5 cost estimates (the ablation arms multiply the public-context series)."""

    def n_series(self, cfg: dict) -> int:
        p = self.p
        if self.exp == "util":
            return p["n"] * (2 + sum(p["xl"]) + len(p.get("arms", ())) * sum(p["d1"]))
        if self.exp == "placement":
            return 0
        return super().n_series(cfg)

    def cost(self, cfg: dict) -> float:
        if self.exp == "placement":
            return 60.0
        if self.exp == "cost":
            return float(cfg["serve"].get("est_sec", 3600))
        return super().cost(cfg)


def build_cells(cfg: dict) -> list:
    cells: list = []
    budget = float(cfg["part_budget_sec"])
    cps = cfg["cost_per_series"]
    for u in cfg["utility"]:
        arms = tuple(u["arms"])
        per = 2 + sum(u["xl_k"]) + len(arms) * sum(u["d1_k"])
        for m in u["models"]:
            for i, n in enumerate(M._parts(u["n_targets"], per, cps[m], budget)):
                cells.append(V5Cell("util", u["dataset"], m, tuple(sorted({
                    "ell": u["ell"], "xl": tuple(u["xl_k"]), "d1": tuple(u["d1_k"]), "d4": False, "arms": arms,
                    "part": i, "n": n}.items()))))
    for f in cfg["attack"]["families"]:
        for ds in f["datasets"]:
            for m in f["models"]:
                for k in f["k"]:
                    sizes = f["sizes"][k] if isinstance(f.get("sizes"), dict) else cfg["attack"]["sizes"][k]
                    for rep in f["reps"]:
                        for role, n_role in zip(("train", "test"), sizes):
                            for i, n in enumerate(M._parts(n_role, k + 1, cps[m], budget)):
                                cells.append(V5Cell("attack", ds, m, tuple(sorted({
                                    "fam": f["fam"], "filler": f.get("filler", "none"), "k": k, "probe": f.get("probe", "short"),
                                    "role": role, "rep": rep, "part": i, "n": n}.items()))))
    for ds in sorted({ds for f in cfg["attack"]["families"] for ds in f["datasets"]}):
        cells.append(V5Cell("ref", ds, "none", ()))  # population references for the attack analysis
    if cfg.get("placement", {}).get("run", False):
        cells.append(V5Cell("placement", "synthetic", "none", ()))
    for m in cfg["serve"].get("models", []):
        cells.append(V5Cell("cost", cfg["serve"]["dataset"], m, ()))
    ids = [c.cell_id for c in cells]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell ids in the configuration")
    return cells


# --------------------------------------------------------------------------- E5a: public-context ablation


def _windows(ds, pub: np.ndarray, pub_valid: np.ndarray, d: int, gap: int, weekday: str) -> tuple[list[int], np.ndarray]:
    """Public users with at least one valid window >= gap days before origin d, on the same / a different / any
    weekday -> (indices into pub, candidate days)."""
    dist = d - np.arange(ds.D)
    past = dist >= gap
    same = ds.weekday == ds.weekday[d]
    ok = past & (same if weekday == "same" else (~same if weekday == "other" else np.ones_like(same)))
    ok_days = np.where(ok)[0]
    return [i for i in range(len(pub)) if pub_valid[i, ok_days].any()], ok_days


def _draw_windows(rng, ds, pub, pub_valid, cand, ok_days, need: int) -> list[np.ndarray]:
    chosen = rng.choice(cand, size=need, replace=False)
    out = []
    for i in chosen:
        dd = ok_days[pub_valid[i, ok_days]]
        out.append(ds.ctx(int(pub[i]), int(rng.choice(dd))))
    return out


def other_domain_windows(env, cell_dataset: str, rng, need: int) -> list[np.ndarray]:
    """``need`` distinct (user, day) 14-day windows of public users of the other dataset (any day)."""
    other = [n for n in env.cfg["datasets"] if n != cell_dataset][0]
    ods = env.ds(other)
    pub = ods.pools["public"]
    valid = ods.valid[pub] & ods.split["any"][None, :]
    pairs = np.argwhere(valid)
    pick = pairs[rng.choice(len(pairs), size=need, replace=False)]
    return [ods.ctx(int(pub[i]), int(d)) for i, d in pick]


def run_util(cell, env, seed: int) -> tuple[dict, dict]:
    cfg, p = env.cfg, cell.p
    ds = env.ds(cell.dataset)
    C, L = ds.C, p["ell"]
    xl, d1, arms = list(p["xl"]), list(p["d1"]), list(p["arms"])
    kmax = max(xl) if xl else 1
    need = max(d1) - 1
    rng = np.random.default_rng(seed)
    gap = int(cfg["d1"]["min_gap_days"])
    pub = ds.pools["public"]
    pub_valid = ds.valid[pub] & ds.split["any"][None, :]
    days = ds.feasible("test", {"target": 1, "service": kmax - 1})
    keep = []
    for d in days:
        if all(len(_windows(ds, pub, pub_valid, int(d), gap, w)[0]) >= need for w in ("same", "other", "any")):
            keep.append(int(d))
    days = np.array(keep, dtype=int)
    if len(days) == 0:
        return M._infeasible("no test origin with enough past public windows on the same and on other weekdays")
    configs = ["iso", "full"] + [f"xl{k}" for k in xl] + [f"{a}_{k}" for a in arms for k in d1]
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
        wins = {}
        for a, wd in (("d1p", "same"), ("d1x", "other"), ("d1r", "any")):
            if a in arms:
                cand, ok_days = _windows(ds, pub, pub_valid, d, gap, wd)
                wins[a] = _draw_windows(rng, ds, pub, pub_valid, cand, ok_days, need)
        if "d1o" in arms:
            wins["d1o"] = other_domain_windows(env, cell.dataset, rng, need)
        for a in arms:
            for k in d1:
                rows[f"{a}_{k}"] = [tgt] + wins[a][: k - 1]
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
               "n_feasible_days": int(len(days)),
               "wql_mean": {c: float(np.nanmean(W[c])) for c in configs},
               "gain_vs_iso": {c: float(1 - np.nanmean(W[c]) / np.nanmean(iso)) for c in configs}}
    return summary, arr


# --------------------------------------------------------------------------- E5d: placement emulation (no model)


def _block_counts(pos_att: np.ndarray, pos_vic: np.ndarray, n_total: int, B: int):
    """For each victim position: attacker members in its block of B consecutive positions and that block's size (the
    last block of the table may be partial)."""
    blk_v = pos_vic // B
    blk_a = np.bincount(pos_att // B, minlength=n_total // B + 1) if len(pos_att) else np.zeros(n_total // B + 1, dtype=int)
    size = np.minimum(B, n_total - blk_v * B)
    return blk_a[blk_v], size


def _victim_stats(a: np.ndarray, size: np.ndarray, B: int) -> dict:
    co = a > 0
    share = a / size
    return {"n": int(len(a)), "co_located": int(co.sum()), "share_if_co": share[co],
            "share_ge_half": int((share >= 0.5).sum()),
            "isolated_255": int((a >= B - 1).sum()),                  # 255 of a full block of 256
            "isolated_any": int(((a == size - 1) & (size > 1)).sum())}  # every other member is the attacker's


def simulate_policy(policy: str, N0: int, lam: float, m: int, B: int, days: int, rng) -> dict:
    """One Monte Carlo run of a nightly table forecast in consecutive blocks of B items.

    ``ordered``: items in registration (ID) order: N0 established clients, then the honest registrations of the horizon
    (Poisson(lam) per day) and the attacker's m meters registered as one contiguous burst at a uniformly random time.
    ``informed``: registration order, but the attacker sees the ID counter: whenever an honest registration opens a
    block (its position is a multiple of B), it registers min(B - 1, remaining budget) meters right after it.
    ``informed_pad``: as informed, but at the start the attacker first registers meters up to the next block boundary
    (if its budget allows), so that the next honest registration opens a block.
    ``hashed``: items in random-ID order; every item's position is uniform.
    Statistics are reported for two victim populations: the new clients of the horizon and all honest clients.
    """
    n_new = int(rng.poisson(lam * days))
    if policy == "hashed":
        n = N0 + n_new + m
        perm = rng.permutation(n)
        pos_att, pos_est, pos_new = perm[:m], perm[m:m + N0], perm[m + N0:]
    elif policy == "ordered":
        k_before = int(rng.integers(0, n_new + 1))  # the burst falls between two honest registrations, uniformly
        pos_new = N0 + np.arange(n_new) + (np.arange(n_new) >= k_before) * m
        pos_att = N0 + k_before + np.arange(m)
        pos_est = np.arange(N0)
        n = N0 + n_new + m
    elif policy in ("informed", "informed_pad"):
        pos, att, new = N0, [], []
        budget = m
        pad = (-N0) % B  # informed_pad: once, at the start, register meters up to the next block boundary
        if policy == "informed_pad" and 0 < pad <= budget:
            att.extend(range(pos, pos + pad))
            pos += pad
            budget -= pad
        for _ in range(n_new):
            new.append(pos)
            opens = pos % B == 0
            pos += 1
            if opens and budget > 0:
                f = min(B - 1, budget)
                att.extend(range(pos, pos + f))
                pos += f
                budget -= f
        pos_att, pos_new, pos_est, n = np.array(att, dtype=int), np.array(new, dtype=int), np.arange(N0), pos
    else:
        raise ValueError(policy)
    pos_att = np.asarray(pos_att, dtype=int)
    out = {"n_new": n_new, "attacker_meters": int(len(pos_att))}
    for name, pv in (("new", np.asarray(pos_new, dtype=int)), ("all", np.concatenate([pos_est, pos_new]).astype(int))):
        a, size = _block_counts(pos_att, pv, n, B)
        out[name] = _victim_stats(a, size, B)
    return out


def simulate_service_policy(N_est: int, k: int, m: int, rng, n_rep: int) -> dict:
    """The paper's service model: each new client is grouped with k - 1 established clients drawn at random. An
    attacker with m new meters reaches a *given* established victim V with probability 1 - (1 - (k-1)/N_est)^m,
    always with occupancy 1/k."""
    p = min(1.0, (k - 1) / N_est)
    hits = rng.binomial(m, p, n_rep) > 0
    return {"p_reach_given_victim": float(hits.mean()), "p_exact": 1 - (1 - p) ** m, "occupancy": 1.0 / k}


def run_placement(cell, env, seed: int) -> tuple[dict, dict]:
    pc = env.cfg["placement"]
    rng = np.random.default_rng(seed)
    B = int(pc["block"])
    rows = []
    for policy in pc["policies"]:
        for N0 in pc["N0"]:
            for lam in pc["lam"]:
                for m in pc["m"]:
                    runs = [simulate_policy(policy, int(N0), float(lam), int(m), B, int(pc["days"]), rng) for _ in range(int(pc["reps"]))]
                    row = {"policy": policy, "N0": int(N0), "lam_per_day": float(lam), "m": int(m), "days": int(pc["days"]),
                           "runs": len(runs), "attacker_meters_used": float(np.mean([r["attacker_meters"] for r in runs]))}
                    for pop in ("new", "all"):
                        st = [r[pop] for r in runs]
                        nv = sum(x["n"] for x in st)
                        shares = np.concatenate([x["share_if_co"] for x in st]) if nv else np.zeros(0)
                        row.update({f"{pop}_victims": nv,
                                    f"{pop}_p_co_located": (sum(x["co_located"] for x in st) / nv) if nv else None,
                                    f"{pop}_co_located_per_run": float(np.mean([x["co_located"] for x in st])),
                                    f"{pop}_share_if_co_median": float(np.median(shares)) if len(shares) else None,
                                    f"{pop}_share_if_co_p95": float(np.quantile(shares, 0.95)) if len(shares) else None,
                                    f"{pop}_share_ge_half_per_run": float(np.mean([x["share_ge_half"] for x in st])),
                                    f"{pop}_isolated_255_per_run": float(np.mean([x["isolated_255"] for x in st])),
                                    f"{pop}_isolated_any_per_run": float(np.mean([x["isolated_any"] for x in st]))})
                    rows.append(row)
    svc = []
    for N_est in pc["service_N_est"]:
        for k in pc["service_k"]:
            for m in pc["m"]:
                svc.append({"N_est": int(N_est), "k": int(k), "m": int(m),
                            **simulate_service_policy(int(N_est), int(k), int(m), rng, int(pc["reps"]) * 50)})
    return {"block": B, "rows": rows, "service_policy": svc}, {}


# --------------------------------------------------------------------------- E5b: end-to-end serving run


def _vmhwm_reset() -> bool:
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def _vmhwm_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def run_serve(cell, env, seed: int) -> tuple[dict, dict]:
    """One nightly run for N new clients on one origin day d, on this (idle) machine, mode by mode."""
    import os

    cfg = env.cfg
    sc = cfg["serve"]
    ds = env.ds(cell.dataset)
    C, L = ds.C, int(sc["ell"])
    threads = min(int(sc["threads"]), os.cpu_count() or 1)
    fc = env.forecaster(cell.model, threads=threads)
    rng = np.random.default_rng(seed)
    k = int(sc["k"])
    P, eps = int(sc["P"]), float(sc["eps"])
    gap = int(cfg["d1"]["min_gap_days"])
    pub = ds.pools["public"]
    pub_valid = ds.valid[pub] & ds.split["any"][None, :]
    days = ds.feasible("test", {"target": 1, "service": k - 1})
    days = np.array([d for d in days if len(_windows(ds, pub, pub_valid, int(d), gap, "same")[0]) >= k - 1], dtype=int)
    d = int(days[len(days) // 2])
    N = int(sc["n_clients"])
    # the new clients: last-24-hour windows of target users (any test day); the night's co-tenants, public windows and
    # release are those of day d
    tdays = ds.feasible("test", {"target": 1})
    news = []
    while len(news) < N:
        dd = int(rng.choice(tdays))
        u = M.pick(rng, ds, "target", dd, 1, None)
        if u is None:
            continue
        w = ds.ctx(int(u[0]), dd)[-L:]
        if np.isfinite(w).sum() == 0:
            continue
        news.append(w.astype(np.float32))
    B = int(sc["series_per_call"])
    from crossleak.baselines import fit_ridge, predict_ridge

    t_fit = time.perf_counter()
    rm = fit_ridge(ds, int(sc["ridge_train"]), seed)
    ridge_fit_s = time.perf_counter() - t_fit
    env.centres(cell.dataset, P)  # one-off public k-means, outside the timed modes
    rows = []

    def chunks(groups):
        out, cur, n = [], [], 0
        for g in groups:
            if cur and n + g.shape[0] > B:
                out.append(cur)
                cur, n = [], 0
            cur.append(g)
            n += g.shape[0]
        if cur:
            out.append(cur)
        return out

    def timed_mode(name: str, build):
        """Times one mode from group construction to the last forecast. Excluded: model load, the ridge fit, the one-off
        public k-means and one untimed warm-up call of the mode's call shape."""
        rss0 = _rss_mb()
        _vmhwm_reset()
        t0 = time.perf_counter()
        groups, extra = build()
        t_prep = time.perf_counter() - t0
        lat, done_at = [], []
        if groups is not None:
            calls = chunks(groups)
            fc.predict_groups(calls[0], max_series_per_call=10 ** 9)  # warm-up (not timed)
            t1 = time.perf_counter()
            for ch in calls:
                s_ = time.perf_counter()
                fc.predict_groups(ch, max_series_per_call=10 ** 9)
                e_ = time.perf_counter()
                lat.append(e_ - s_)
                done_at.extend([t_prep + (e_ - t1)] * len(ch))
            t_fc = time.perf_counter() - t1
        else:
            t_fc = 0.0
            done_at = [t_prep] * N
        total = t_prep + t_fc
        peak = _vmhwm_mb()
        rows.append({"mode": name, "n_clients": N, "series_per_client": (groups[0].shape[0] if groups else 0),
                     "calls": len(lat), "prep_s": t_prep, "forecast_s": t_fc, "total_s": total,
                     "call_p50_s": float(np.median(lat)) if lat else None, "call_p95_s": float(np.quantile(lat, 0.95)) if lat else None,
                     "client_done_p50_s": float(np.median(done_at)), "client_done_p95_s": float(np.quantile(done_at, 0.95)),
                     "throughput_clients_per_s": (N / total) if total > 0 else None,
                     "rss_start_mb": rss0, "peak_rss_mb": peak, "peak_rss_increase_mb": peak - rss0, **extra})
        log.info("serve %s: %.1fs total (%d calls)", name, total, len(lat))

    def b_d0():
        return [pad_left(w, C)[None] for w in news], {}

    def b_xl():
        est = ds.users("service", d)
        return [np.stack([pad_left(w, C)] + [ds.ctx(int(c), d) for c in rng.choice(est, size=k - 1, replace=False)]) for w in news], {}

    def b_d1():
        cand, ok_days = _windows(ds, pub, pub_valid, d, gap, "same")
        out = []
        for w in news:
            out.append(np.stack([pad_left(w, C)] + _draw_windows(rng, ds, pub, pub_valid, cand, ok_days, k - 1)))
        return out, {}

    def b_d4():
        t = time.perf_counter()
        kind = cfg["d4"]["representation"]
        users = ds.users("service", d)
        R = np.stack([dp.represent(ds.ctx(int(u), d), kind) for u in users])
        rel = dp.release(R, env.centres(cell.dataset, P), eps, float(cfg["delta"]), dp.default_clip(kind, C),
                         np.random.default_rng(stable_seed("serve_release", str(d), base=cfg["seed"])), float(cfg["d4"]["sum_share"]))
        protos = [dp.to_series(r, kind, C) for r in rel.prototypes]
        t_rel = time.perf_counter() - t
        return [np.stack([pad_left(w, C)] + protos) for w in news], {"release_s": t_rel, "release_users": int(len(users))}

    def b_snaive():
        _ = [np.repeat(w[-24:][None], 9, axis=0) for w in news]
        return None, {}

    def b_ridge():
        wd = int(ds.weekday[d])
        _ = [predict_ridge(rm, w[-24:], wd) for w in news]
        return None, {"ridge_scaling": rm["scaling"], "ridge_alpha": rm["alpha"]}

    builders = {"snaive": b_snaive, "ridge": b_ridge, "d0": b_d0, "xl": b_xl, "d1p": b_d1, "d4": b_d4}
    for name in sc["modes"]:
        timed_mode(name, builders[name])
    summary = {"threads": threads, "origin_day": d, "origin_date": str(ds.val.days[d].date()), "k": k, "P": P, "eps": eps,
               "series_per_call": B, "rows": rows, "rss_reset_supported": _vmhwm_reset(), "cpu_count": os.cpu_count(),
               "ridge_fit_s": ridge_fit_s, "ridge_scaling": rm["scaling"], "ridge_unique_windows": rm.get("n_unique"),
               "timing_excludes": "model load, ridge fit, public k-means, one warm-up call per mode",
               "peak_rss_note": "process peak during the mode (VmHWM reset at the mode start; memory kept from earlier modes counts)"}
    return summary, {}


# --------------------------------------------------------------------------- install


def _short(cell, s: dict) -> str:
    if s.get("status") == "infeasible":
        return f"infeasible: {s.get('reason')}"
    if cell.exp == "cost":
        return "; ".join(f"{r['mode']}:{r['total_s']:.0f}s" for r in s.get("rows", []))
    if cell.exp == "placement":
        return f"{len(s.get('rows', []))} policy rows"
    return _ORIGINAL["short"](cell, s)


def install() -> None:
    """Route the main driver's cells and runners through this module (idempotent)."""
    M.build_cells = build_cells
    M._short = _short
    M.RUNNERS["util"] = run_util
    M.RUNNERS["placement"] = run_placement
    M.RUNNERS["cost"] = run_serve


def uninstall() -> None:
    if M.build_cells is build_cells:
        M.build_cells = _ORIGINAL["build_cells"]
    M._short = _ORIGINAL["short"]
    if M.RUNNERS["util"] is run_util:
        M.RUNNERS["util"] = _ORIGINAL["util"]
    M.RUNNERS["cost"] = _ORIGINAL["cost"]
    if _ORIGINAL["placement"] is None:
        M.RUNNERS.pop("placement", None)
    else:
        M.RUNNERS["placement"] = _ORIGINAL["placement"]

