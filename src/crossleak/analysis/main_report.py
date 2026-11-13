"""Main-study report: estimands, tables and the pre-registered verdicts (prereg/MAIN_PREREG.md).

Reads the run directories of the main experiment (one per machine, or unpacked bundles), merges the parts of each
configuration, fits the attacker's decoders (``decoders``: "none" | "linear" | "full"), and writes
reports/<experiment>/: MAIN_REPORT.md, main_summary.json, utility.csv, attacks.csv, channel.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossleak.config import TAG, repo_root, reports_dir, sha256_file

log = logging.getLogger("crossleak.main_report")

# ---- pre-registered constants (prereg/MAIN_PREREG.md section 5). Do not edit after the freeze.
ALPHA = 0.05
GAIN_YES = 0.05
HISTORY_NO_BENEFIT = 0.02
CL3_AUC = 0.65
CL4_DELTA_R = 0.10
CL5_AUC = 0.60
CL5_DECAY = 0.25
CL6_RETAINED = 0.50
CL6_EPS = (1.0, 2.0, 4.0)
CL8_SECONDS_PER_1000 = 900.0
CL9_TOP1 = 0.15
INVARIANCE_TOL = 1e-3
CHANNEL_FACTOR = 100.0
DPAUDIT_TOL = 0.03
INVALID_MAX = 0.10
PRIMARY_MODEL = "c2base"
SYBIL_MODEL = "c2small"
PRIMARY_PROBE = "short"


def check_freeze(name: str) -> dict:
    path = repo_root() / "prereg" / name
    if not path.exists():
        return {"status": "missing", "files": {}}
    files, ok = {}, True
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        f = repo_root() / rel.strip()
        actual = sha256_file(f) if f.exists() else None
        files[rel.strip()] = {"expected": digest, "actual": actual, "match": actual == digest}
        ok &= actual == digest
    return {"status": "match" if ok else "MODIFIED", "files": files}


# --------------------------------------------------------------------------- loading


def load_results(roots: list[Path]) -> tuple[dict[str, dict], list[dict]]:
    results: dict[str, dict] = {}
    failed = []
    for root in roots:
        for f in sorted(Path(root).rglob("result.json")):
            r = json.loads(f.read_text())
            r["_dir"] = str(f.parent)
            results.setdefault(r["cell_id"], r)
        for f in sorted(Path(root).rglob("FAILED")):
            if (f.parent / "DONE").exists():
                continue
            info = json.loads(f.read_text())
            info["cell_id"] = f.parent.name
            failed.append(info)
    failed = [f for f in failed if f["cell_id"] not in results]
    return results, failed


def arrays(r: dict) -> dict:
    p = Path(r["_dir"]) / "arrays.npz"
    return dict(np.load(p, allow_pickle=False)) if p.exists() else {}


def _boot(stat, clusters, n_boot, seed, alpha=ALPHA):
    from crossleak.metrics import cluster_bootstrap

    return cluster_bootstrap(stat, clusters, n_boot, seed, alpha)


def _boot2(statA, cA, statB, cB, combine, n_boot, seed, alpha=ALPHA) -> dict:
    """Percentile CI of combine(statA, statB) for two independent samples, each resampled by cluster."""
    rng = np.random.default_rng(seed)

    def members(c):
        u, inv = np.unique(c, return_inverse=True)
        return [np.where(inv == i)[0] for i in range(len(u))]

    mA, mB = members(cA), members(cB)
    est = combine(statA(np.arange(len(cA))), statB(np.arange(len(cB))))
    vals = []
    for _ in range(n_boot):
        ia = np.concatenate([mA[i] for i in rng.integers(0, len(mA), len(mA))])
        ib = np.concatenate([mB[i] for i in rng.integers(0, len(mB), len(mB))])
        v = combine(statA(ia), statB(ib))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {"est": float(est), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"est": float(est), "lo": float(lo), "hi": float(hi)}


# --------------------------------------------------------------------------- utility


def analyse_utility(results: dict, cfg: dict, n_boot: int) -> dict:
    groups: dict = defaultdict(list)
    for r in results.values():
        if r["exp"] == "util" and r["status"] == "ok":
            groups[(r["dataset"], r["model"], int(r["params"]["ell"]))].append(r)
    out: dict = {}
    for (ds, model, ell), rs in sorted(groups.items()):
        arrs = [arrays(r) for r in sorted(rs, key=lambda r: r["params"]["part"])]
        keys = [k for k in arrs[0] if k.startswith("wql_")]
        W = {k[4:]: np.concatenate([a[k] for a in arrs]) for k in keys}
        users = np.concatenate([a["user"] for a in arrs])
        iso = W["iso"]
        xl = sorted(int(c[2:]) for c in W if c.startswith("xl"))
        ref = f"xl{cfg['xl_ref'][ds]}"
        seed = 1000 + ell
        res: dict = {"n_targets": int(len(users)), "n_users": int(len(np.unique(users))), "configs": {}}
        for c, w in W.items():
            if c == "iso":
                continue
            ok = np.isfinite(w) & np.isfinite(iso)
            uu = users[ok]
            wc, wi = w[ok], iso[ok]
            g = _boot(lambda i: 1.0 - wc[i].mean() / wi[i].mean(), uu, n_boot, seed)
            row = {"wql_mean": float(wc.mean()), "gain": g, "share_improved": float((wc < wi).mean())}
            if c.startswith("xl") and len(xl) > 1:
                row["gain_adj"] = _boot(lambda i: 1.0 - wc[i].mean() / wi[i].mean(), uu, n_boot, seed, ALPHA / len(xl))
            if (c.startswith("d4") or c.startswith("d1")) and ref in W:
                wr = W[ref][ok]
                okr = np.isfinite(wr)
                row["retained_vs_" + ref] = _boot(
                    lambda i: (1.0 - wc[i][okr[i]].mean() / wi[i][okr[i]].mean()) / (1.0 - wr[i][okr[i]].mean() / wi[i][okr[i]].mean()),
                    uu, n_boot, seed + 1)
            if c.startswith("d4"):
                eps = float(c.split("eps")[1])
                if eps in CL6_EPS:
                    row["gain_adj"] = _boot(lambda i: 1.0 - wc[i].mean() / wi[i].mean(), uu, n_boot, seed, ALPHA / len(CL6_EPS))
            res["configs"][c] = row
        res["wql_iso_mean"] = float(np.nanmean(iso))
        out.setdefault(ds, {}).setdefault(model, {})[ell] = res
    return out


# --------------------------------------------------------------------------- channel


def analyse_channel(results: dict, n_boot: int) -> dict:
    fam: dict = defaultdict(list)
    for r in results.values():
        if r["exp"] == "channel" and r["status"] == "ok":
            key = (r["dataset"], r["model"], r["params"]["fam"], r["params"]["filler"], int(r["params"]["k"]), r["params"]["probe"])
            fam[key].append(r)
    data = {}
    for key, rs in fam.items():
        arrs = [arrays(r) for r in rs]
        data[key] = (np.concatenate([a["i_swap"] for a in arrs]), np.concatenate([a["victim"] for a in arrs]))
    out = {}
    for key, (x, v) in sorted(data.items()):
        ds, model, f, filler, k, probe = key
        row = {"n_pairs": int(len(x)), "i_swap_median": _boot(lambda i: float(np.median(x[i])), v, n_boot, 7)}
        base = data.get((ds, model, "single", "none", 2, probe))
        if base is not None:
            xb, vb = base
            row["ratio_to_single_k2"] = _boot2(lambda i: float(np.median(x[i])), v, lambda i: float(np.median(xb[i])), vb,
                                               lambda a, b: a / b if b > 0 else float("nan"), n_boot, 8)
        out["__".join([ds, model, f, filler, f"k{k}", probe])] = row
    return out


# --------------------------------------------------------------------------- attacks


def _ref(results: dict, ds: str) -> dict | None:
    r = results.get(f"ref__{ds}__none")
    return arrays(r) if r and r["status"] == "ok" else None


def analyse_attack_family(key: str, train: list[dict], test: list[dict], ref: dict, decoders: str, n_boot: int, seed: int) -> dict:
    from crossleak import attack as A
    from crossleak.decode import evening_score
    from crossleak.metrics import auc

    tr = [arrays(r) for r in train]
    te = [arrays(r) for r in test]
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])  # noqa: E731
    keys = ("fxl", "fiso", "cal", "Y", "dm", "ev14", "victim", "attacker", "day")
    Ttr = {k: cat(tr, k) for k in keys}
    Tte = {k: cat(te, k) for k in keys}
    Xtr, Ctr = A.features(Ttr)
    Xte, Cte = A.features(Tte)
    thr = float(np.median(evening_score(Ttr["Y"])))
    Ltr = A.labels(Ttr["Y"], Ttr["dm"], Ttr["day"], ref["ref_even_med_att"], thr, Ttr["ev14"], ref["ref_resid_med_att"])  # attacker-accessible users only
    Lte = A.labels(Tte["Y"], Tte["dm"], Tte["day"], ref["ref_even_med"], thr, Tte["ev14"], ref["ref_resid_med"])
    vic = Tte["victim"]
    out: dict = {"n_train": int(len(Xtr)), "n_test": int(len(Xte)), "n_test_victims": int(len(np.unique(vic))),
                 "label_rate_test": {k: float((v[v >= 0]).mean()) for k, v in Lte.items()}}
    dx = A.linear_decoder(Xtr, Ttr["Y"], Ltr, Xte, seed)
    dc = A.linear_decoder(Ctr, Ttr["Y"], Ltr, Cte, seed)
    perm = np.random.default_rng(seed).permutation(len(Xtr))
    dp_ = A.linear_decoder(Xtr, Ttr["Y"][perm], {"L1": Ltr["L1"][perm]}, Xte, seed)
    out["ridge_alpha"] = dx["ridge_alpha"]
    for lab in A.LABELS:
        if lab not in dx["scores"]:
            continue
        y = Lte[lab]
        ok = y >= 0
        sx, sc = dx["scores"][lab][ok], dc["scores"][lab][ok]
        yy, vv = y[ok], vic[ok]
        out[f"auc_{lab}"] = _boot(lambda i: auc(yy[i], sx[i]), vv, n_boot, seed + 1)
        out[f"auc_{lab}_ctrl"] = auc(yy, sc)
        out[f"delta_auc_{lab}"] = _boot(lambda i: auc(yy[i], sx[i]) - auc(yy[i], sc[i]), vv, n_boot, seed + 2)
    out["auc_L1_perm"] = auc(Lte["L1"], dp_["scores"]["L1"])
    rm = ref["ref_mean"][Tte["day"]]
    ybar = Ttr["Y"].mean(axis=0, keepdims=True)
    r1x, r1c, r1p = (A.within_day_r(d["yhat"], Tte["Y"], rm) for d in (dx, dc, dp_))
    from crossleak.metrics import rowwise_pearson

    r2x, r2c = rowwise_pearson(dx["yhat"] - ybar, Tte["Y"] - ybar), rowwise_pearson(dc["yhat"] - ybar, Tte["Y"] - ybar)
    ok = np.isfinite(r1x) & np.isfinite(r1c)
    a, b, vv = r1x[ok], r1c[ok], vic[ok]
    out["r1_median"] = float(np.nanmedian(r1x))
    out["r1_median_ctrl"] = float(np.nanmedian(r1c))
    out["r1_median_perm"] = float(np.nanmedian(r1p))
    out["delta_r1"] = _boot(lambda i: float(np.mean(a[i] - b[i])), vv, n_boot, seed + 3)
    out["r2_median"] = float(np.nanmedian(r2x))
    out["r2_median_ctrl"] = float(np.nanmedian(r2c))
    cand = ref["cand"].astype(np.float32)
    pool = ref["pools_victim"]
    for N in (10, 50):
        lk = A.linkage(dx["yhat"], vic, Tte["day"], cand, pool, N, seed + 4)
        lc = A.linkage(dc["yhat"], vic, Tte["day"], cand, pool, N, seed + 4)
        okk = np.isfinite(lk["top1"])
        if okk.sum() < 50:
            continue
        t1, vv = lk["top1"][okk], vic[okk]
        out[f"a1_N{N}"] = {"n": int(okk.sum()), "chance_top1": 1.0 / N, "top1": _boot(lambda i: float(t1[i].mean()), vv, n_boot, seed + 5),
                           "top5": float(np.nanmean(lk["top5"])), "top1_ctrl": float(np.nanmean(lc["top1"])), "top5_ctrl": float(np.nanmean(lc["top5"]))}
    if decoders == "full":
        try:  # an estimand only: its failure must not drop the primary (linear) results of this configuration
            nx = A.nonlinear_decoder(Xtr, Ttr["Y"], Ltr, Xte, seed)
            y = Lte["L1"]
            sx, sc = nx["scores"]["L1"], dc["scores"]["L1"]
            out["nl_auc_L1"] = _boot(lambda i: auc(y[i], sx[i]), vic, n_boot, seed + 6)
            out["nl_delta_auc_L1"] = _boot(lambda i: auc(y[i], sx[i]) - auc(y[i], sc[i]), vic, n_boot, seed + 7)
            r1n = A.within_day_r(nx["yhat"], Tte["Y"], rm)
            out["nl_r1_median"] = float(np.nanmedian(r1n))
        except Exception as exc:  # noqa: BLE001
            out["nl_error"] = repr(exc)[:200]
    return out


def analyse_attacks(results: dict, decoders: str, n_boot: int, n_jobs: int) -> dict:
    if decoders == "none":
        return {}
    fams: dict = defaultdict(lambda: {"train": [], "test": []})
    for r in results.values():
        if r["exp"] == "attack" and r["status"] == "ok":
            fams[r["family_key"]][r["params"]["role"]].append(r)
    refs = {}
    jobs = []
    for i, (key, rr) in enumerate(sorted(fams.items())):
        if not rr["train"] or not rr["test"]:
            continue
        ds = rr["test"][0]["dataset"]
        if ds not in refs:
            refs[ds] = _ref(results, ds)
        if refs[ds] is None:
            continue
        jobs.append((key, sorted(rr["train"], key=lambda r: r["params"]["part"]), sorted(rr["test"], key=lambda r: r["params"]["part"]), ds))
    t0 = time.time()

    def one(job):
        key, tr, te, ds = job
        try:
            return key, analyse_attack_family(key, tr, te, refs[ds], decoders, n_boot, 4242)
        except Exception as exc:  # noqa: BLE001
            log.exception("attack analysis failed for %s", key)
            return key, {"error": repr(exc)}

    if n_jobs > 1 and len(jobs) > 1:
        from joblib import Parallel, delayed

        res = Parallel(n_jobs=n_jobs)(delayed(one)(j) for j in jobs)
    else:
        res = [one(j) for j in jobs]
    out = {}
    for (key, r), job in zip(res, jobs):
        p = job[2][0]["params"]
        r.update({"dataset": job[3], "model": job[2][0]["model"], "fam": p["fam"], "filler": p["filler"], "k": int(p["k"]), "probe": p["probe"]})
        out[key] = r
    log.info("decoders (%s) for %d attack configurations in %.1f min", decoders, len(out), (time.time() - t0) / 60)
    return out


# --------------------------------------------------------------------------- verdicts


def _find_attack(att: dict, ds: str, model: str, fam: str, filler: str, k: int, probe: str = PRIMARY_PROBE) -> dict | None:
    for v in att.values():
        if (v.get("dataset"), v.get("model"), v.get("fam"), v.get("filler"), v.get("k"), v.get("probe")) == (ds, model, fam, filler, k, probe):
            return v if "error" not in v else None
    return None


def verdicts(cfg: dict, util: dict, chan: dict, att: dict, inv: dict, dpa: dict, cost: dict, coverage: dict, decoders: str) -> dict:
    V: dict = {}
    datasets = list(cfg["datasets"])
    # CL1 gain with one day of history; CL1b no benefit with three days
    for ds in datasets:
        u = util.get(ds, {}).get(PRIMARY_MODEL, {})
        r24, r72 = u.get(24), u.get(72)
        if r24:
            xl = {c: v for c, v in r24["configs"].items() if c.startswith("xl")}
            yes = any(v["gain"]["est"] >= GAIN_YES and v.get("gain_adj", v["gain"])["lo"] > 0 for v in xl.values())
            weak = any(v.get("gain_adj", v["gain"])["lo"] > 0 for v in xl.values())
            V[f"CL1_{ds}"] = "YES" if yes else ("WEAK" if weak else "NO")
        else:
            V[f"CL1_{ds}"] = "NO_DATA"
        if r72:
            xl = {c: v for c, v in r72["configs"].items() if c.startswith("xl")}
            V[f"CL1b_{ds}"] = "NO_BENEFIT" if all(v.get("gain_adj", v["gain"])["hi"] < HISTORY_NO_BENEFIT for v in xl.values()) else "BENEFIT"
        else:
            V[f"CL1b_{ds}"] = "NO_DATA"
    # CL2 channel well above numerical noise; invariance exact
    for ds in datasets:
        ok, seen = True, False
        for m in cfg["models"]:
            c = chan.get(f"{ds}__{m}__single__none__k2__{PRIMARY_PROBE}")
            i = inv.get(f"{ds}__{m}")
            if c is None or i is None:
                continue
            seen = True
            ok &= i["probe_max_abs_change"] < INVARIANCE_TOL and c["i_swap_median"]["est"] >= CHANNEL_FACTOR * max(i["probe_max_abs_change"], 1e-9)
        V[f"CL2_{ds}"] = ("YES" if ok else "NO") if seen else "NO_DATA"
    if decoders != "none":
        for ds in datasets:
            a2 = _find_attack(att, ds, PRIMARY_MODEL, "single", "none", 2)
            if a2 and "auc_L1" in a2:
                ok = a2["auc_L1"]["est"] >= CL3_AUC and a2["delta_auc_L1"]["lo"] > 0
                V[f"CL3_{ds}"] = "YES" if ok else "NO"
                V[f"CL4_{ds}"] = "YES" if (a2["delta_r1"]["est"] >= CL4_DELTA_R and a2["delta_r1"]["lo"] > 0) else "NO"
                a1 = a2.get("a1_N10")
                V[f"CL9_{ds}"] = ("YES" if (a1["top1"]["est"] >= CL9_TOP1 and a1["top1"]["lo"] > a1["chance_top1"]) else "NO") if a1 else "NO_DATA"
            else:
                V[f"CL3_{ds}"] = V[f"CL4_{ds}"] = V[f"CL9_{ds}"] = "NO_DATA"
            a4 = _find_attack(att, ds, PRIMARY_MODEL, "single", "none", 4)
            V[f"CL3_k4_{ds}"] = ("PERSISTS" if a4["delta_auc_L1"]["lo"] > 0 else "NOT_DETECTED") if a4 and "delta_auc_L1" in a4 else "NO_DATA"
            # CL5: sybils defeat large groups
            kmax = max(int(k) for f in cfg["channel"]["families"] if f["fam"] == "single" and ds in f["datasets"] for k in f["k"])
            dec = chan.get(f"{ds}__{SYBIL_MODEL}__single__none__k{kmax}__{PRIMARY_PROBE}", {}).get("ratio_to_single_k2")
            syb = _find_attack(att, ds, SYBIL_MODEL, "sybil", "copy", 256)
            if dec is None or syb is None or "auc_L1" not in syb:
                V[f"CL5_{ds}"] = "NO_DATA"
            else:
                b = syb["auc_L1"]["est"] >= CL5_AUC and syb["delta_auc_L1"]["lo"] > 0
                a = dec["hi"] < CL5_DECAY
                V[f"CL5_{ds}"] = "YES" if (a and b) else ("SYBIL_ONLY" if b else "NO")
            sb = _find_attack(att, ds, PRIMARY_MODEL, "sybil", "copy", 16)
            V[f"CL5_base_k16_{ds}"] = ("PERSISTS" if sb["delta_auc_L1"]["lo"] > 0 else "NOT_DETECTED") if sb and "delta_auc_L1" in sb else "NO_DATA"
    # CL6 D4 keeps >= 50 % of the gain at some eps <= 4 (primary P), ELD primary, LifeSnaps reported
    P = cfg["d4"]["P_primary"]
    for ds in datasets:
        r24 = util.get(ds, {}).get(PRIMARY_MODEL, {}).get(24)
        if not r24:
            V[f"CL6_{ds}"] = "NO_DATA"
            continue
        passing = []
        for e in CL6_EPS:
            from crossleak.experiments.main import d4_name

            c = r24["configs"].get(d4_name(P, e))
            if c is None:
                continue
            ret = c.get(f"retained_vs_xl{cfg['xl_ref'][ds]}")
            if ret and ret["est"] >= CL6_RETAINED and c.get("gain_adj", c["gain"])["lo"] > 0:
                passing.append(e)
        V[f"CL6_{ds}"] = f"YES_eps{min(passing):g}" if passing else "NO"
    # DP audit consistency (sanity: no empirical AUC above the DP ceiling)
    for ds, d in dpa.items():
        viol = [r["eps"] for r in d["rows"] for lab in ("L1", "L2", "L3")
                if r.get(f"auc_{lab}") is not None and r["auc_" + lab] > r["ceiling_attribute_auc"] + DPAUDIT_TOL]
        V[f"DPAUDIT_{ds}"] = "CONSISTENT" if not viol else f"VIOLATION_eps{viol}"
    # CL8 edge cost: 1,000 users per day with D4 (primary P, base model) within 15 min on one 4-vCPU VM
    cb = cost.get(PRIMARY_MODEL)
    if cb:
        row = next((r for r in cb["rows"] if r["mode"] == "d4_batch" and r["group_size"] == P + 1), None)
        if row:
            t = 1000 * row["per_user_p50_s"] + cb["release_1000_users_s"]
            V["CL8"] = "YES" if t <= CL8_SECONDS_PER_1000 else "NO"
            V["CL8_seconds_per_1000_users"] = round(t, 1)
        else:
            V["CL8"] = "NO_DATA"
    else:
        V["CL8"] = "NO_DATA"
    if coverage.get("config_mismatch", 0) > 0:
        V["overall"] = "CONFIG_MISMATCH"  # results produced under another configuration must not be mixed in
    elif coverage["invalid_share"] > INVALID_MAX:
        V["overall"] = "NOT_INTERPRETABLE"
    else:
        V["overall"] = "COMPLETE" if coverage["missing"] == 0 else "PARTIAL"
    if coverage["missing"] > 0:
        for k in list(V):
            if k.startswith("CL") and not str(V[k]).startswith("PARTIAL_") and not k.endswith("seconds_per_1000_users"):
                V[k] = "PARTIAL_" + str(V[k])
    return V


# --------------------------------------------------------------------------- report


def _f(x, nd=3):
    if x is None:
        return "–"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "nan"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(b: dict | None, nd=3) -> str:
    if not b:
        return "–"
    return f"{_f(b.get('est'), nd)} [{_f(b.get('lo'), nd)}, {_f(b.get('hi'), nd)}]"


def build_report(roots: list[Path], cfg: dict, decoders: str = "linear", n_jobs: int = 1, out_dir: Path | None = None) -> Path:
    from crossleak.experiments.main import build_cells

    t0 = time.time()
    exp = cfg["experiment"]
    results, failed = load_results(roots)
    planned = {c.cell_id for c in build_cells(cfg)}
    ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
    infeasible = sorted(k for k, r in results.items() if r.get("status") == "infeasible")
    missing = sorted(planned - set(results))
    mismatch = sorted(k for k, r in results.items() if cfg.get("_sha256") and r.get("config_sha256") != cfg["_sha256"])
    coverage = {"planned": len(planned), "ok": len(ok), "infeasible": len(infeasible), "failed": len(failed),
                "missing": len(missing), "invalid_share": (len(failed) + len(infeasible)) / max(1, len(planned)),
                "config_mismatch": len(mismatch), "config_mismatch_cells": mismatch[:200],
                "infeasible_cells": infeasible, "failed_cells": [f["cell_id"] for f in failed], "missing_cells": missing[:200]}
    n_boot = int(cfg["bootstrap"])
    util = analyse_utility(results, cfg, n_boot)
    chan = analyse_channel(results, n_boot)
    inv = {f"{r['dataset']}__{r['model']}": r["summary"] for r in ok.values() if r["exp"] == "invariance"}
    dpa = {r["dataset"]: r["summary"] for r in ok.values() if r["exp"] == "dpaudit"}
    cost = {r["model"]: r["summary"] for r in ok.values() if r["exp"] == "cost"}
    att = analyse_attacks(results, decoders, n_boot, n_jobs)
    V = verdicts(cfg, util, chan, att, inv, dpa, cost, coverage, decoders)
    summary = {"version": TAG, "experiment": exp, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "decoders": decoders,
               "freeze_main": check_freeze("MAIN_FREEZE.sha256"), "freeze_e0": check_freeze("E0_FREEZE.sha256")["status"],
               "coverage": coverage, "verdicts": V, "utility": util, "channel": chan, "attacks": att, "invariance": inv,
               "dpaudit": dpa, "cost": cost, "runtime_min": round((time.time() - t0) / 60, 1)}
    out = out_dir or (reports_dir() / exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "main_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    _write_csvs(out, util, chan, att)
    md = _markdown(summary, cfg)
    (out / "MAIN_REPORT.md").write_text(md)
    log.info("report written to %s (%.1f min)", out, (time.time() - t0) / 60)
    return out / "MAIN_REPORT.md"


def _write_csvs(out: Path, util: dict, chan: dict, att: dict) -> None:
    with open(out / "utility.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "model", "ell", "config", "n_targets", "wql_mean", "gain", "gain_lo", "gain_hi", "retained", "retained_lo", "retained_hi"])
        for ds, mm in util.items():
            for m, ee in mm.items():
                for ell, r in ee.items():
                    for c, v in r["configs"].items():
                        ret = next((x for k, x in v.items() if k.startswith("retained_vs")), {}) or {}
                        w.writerow([ds, m, ell, c, r["n_targets"], v["wql_mean"], v["gain"]["est"], v["gain"]["lo"], v["gain"]["hi"],
                                    ret.get("est"), ret.get("lo"), ret.get("hi")])
    with open(out / "channel.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "n_pairs", "i_swap_median", "lo", "hi", "ratio_to_k2", "ratio_lo", "ratio_hi"])
        for k, v in chan.items():
            r = v.get("ratio_to_single_k2", {})
            w.writerow([k, v["n_pairs"], v["i_swap_median"]["est"], v["i_swap_median"]["lo"], v["i_swap_median"]["hi"], r.get("est"), r.get("lo"), r.get("hi")])
    with open(out / "attacks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        cols = ["dataset", "model", "fam", "filler", "k", "probe", "n_train", "n_test", "auc_L1", "auc_L1_lo", "auc_L1_hi", "auc_L1_ctrl",
                "auc_L1_perm", "delta_auc_L1", "delta_lo", "delta_hi", "auc_L2", "auc_L3", "auc_L4", "delta_auc_L4", "auc_L5", "delta_auc_L5",
                "r1_median", "r1_median_ctrl", "delta_r1",
                "a1_top1_N10", "a1_top1_N10_ctrl", "nl_auc_L1"]
        w.writerow(cols)
        for v in att.values():
            if "error" in v:
                continue
            g = lambda k, f="est": (v.get(k) or {}).get(f) if isinstance(v.get(k), dict) else v.get(k)  # noqa: E731
            w.writerow([v["dataset"], v["model"], v["fam"], v["filler"], v["k"], v["probe"], v["n_train"], v["n_test"],
                        g("auc_L1"), g("auc_L1", "lo"), g("auc_L1", "hi"), v.get("auc_L1_ctrl"), v.get("auc_L1_perm"), g("delta_auc_L1"),
                        g("delta_auc_L1", "lo"), g("delta_auc_L1", "hi"), g("auc_L2"), g("auc_L3"), g("auc_L4"), g("delta_auc_L4"), g("auc_L5"),
                        g("delta_auc_L5"), v.get("r1_median"), v.get("r1_median_ctrl"),
                        g("delta_r1"), (v.get("a1_N10") or {}).get("top1", {}).get("est"), (v.get("a1_N10") or {}).get("top1_ctrl"), g("nl_auc_L1")])


def _markdown(s: dict, cfg: dict) -> str:
    L = [f"# CrossLeak main study report ({s['version']}, {s['created']})", ""]
    cov = s["coverage"]
    L += [f"Freeze (main): **{s['freeze_main']['status']}**; E0 freeze: {s['freeze_e0']}. Decoders: {s['decoders']}.",
          f"Cells: {cov['ok']} ok, {cov['infeasible']} infeasible, {cov['failed']} failed, {cov['missing']} missing of {cov['planned']} planned "
          f"(invalid share {cov['invalid_share']:.3f}; rule: <= {INVALID_MAX}); results under another config: {cov.get('config_mismatch', 0)}.",
          "", "## Verdicts", "", "| Rule | Verdict |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["verdicts"].items()]
    L += ["", "## Utility (gain vs isolated, 95 % CI; retained = gain / gain of the cross-learning reference)", ""]
    for ds, mm in s["utility"].items():
        for m, ee in mm.items():
            for ell, r in sorted(ee.items()):
                L += [f"**{ds} · {m} · history {ell} h** (n = {r['n_targets']} targets, {r['n_users']} users; isolated WQL {_f(r['wql_iso_mean'], 4)})", "",
                      "| Config | WQL | Gain | Retained | Improved |", "|---|---|---|---|---|"]
                for c, v in r["configs"].items():
                    ret = next((x for k, x in v.items() if k.startswith("retained_vs")), None)
                    L.append(f"| {c} | {_f(v['wql_mean'], 4)} | {_ci(v['gain'])} | {_ci(ret, 2)} | {_f(v['share_improved'], 2)} |")
                L.append("")
    L += ["## Channel (median swap influence in model space; ratio to the single member at k = 2)", "", "| Config | Pairs | Influence | Ratio |", "|---|---|---|---|"]
    for k, v in s["channel"].items():
        L.append(f"| {k} | {v['n_pairs']} | {_ci(v['i_swap_median'], 4)} | {_ci(v.get('ratio_to_single_k2'), 3)} |")
    if s["attacks"]:
        L += ["", "## Attacks (linear decoder; L1 = within-day evening label; R1 = within-day anomaly correlation)", "",
              "| Dataset | Model | Family | Filler | k | Probe | AUC L1 | ctrl | perm | ΔAUC L1 | AUC L2 | AUC L3 | ΔAUC L4 | ΔAUC L5 | r1 | r1 ctrl | Δr1 | A1 top-1 (N=10) | NL AUC L1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in sorted(s["attacks"].values(), key=lambda v: (v.get("dataset", ""), v.get("model", ""), v.get("fam", ""), v.get("filler", ""), v.get("k", 0), v.get("probe", ""))):
            if "error" in v:
                L.append(f"| {v.get('dataset')} | {v.get('model')} | {v.get('fam')} | {v.get('filler')} | {v.get('k')} | {v.get('probe')} | ERROR {v['error'][:60]} |")
                continue
            a1 = v.get("a1_N10")
            L.append(f"| {v['dataset']} | {v['model']} | {v['fam']} | {v['filler']} | {v['k']} | {v['probe']} | {_ci(v.get('auc_L1'))} | {_f(v.get('auc_L1_ctrl'))} | "
                     f"{_f(v.get('auc_L1_perm'))} | {_ci(v.get('delta_auc_L1'))} | {_f((v.get('auc_L2') or {}).get('est'))} | {_f((v.get('auc_L3') or {}).get('est'))} | "
                     f"{_ci(v.get('delta_auc_L4'))} | {_ci(v.get('delta_auc_L5'))} | "
                     f"{_f(v.get('r1_median'))} | {_f(v.get('r1_median_ctrl'))} | {_ci(v.get('delta_r1'))} | "
                     f"{_ci(a1['top1']) if a1 else '–'} | {_ci(v.get('nl_auc_L1'))} |")
    L += ["", "## Invariance", ""] + [f"- {k}: max probe change {_f(v['probe_max_abs_change'], 8)}, victim equivariance error {_f(v['victim_equivariance_max_rel_err'], 8)}" for k, v in s["invariance"].items()]
    L += ["", "## DP audit (adversary knowing every other user; AUC vs the attribute-inference ceiling)", ""]
    for ds, d in s["dpaudit"].items():
        L += [f"**{ds}** ({d['representation']}, clip {_f(d['clip'], 2)}, n_test {d['n_test']}):", "",
              "| eps | noise/coord | AUC L1 | AUC L2 | AUC L3 | AUC L4 | AUC L5 | ceiling |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {r['eps']} | {_f(r['noise_std_per_coordinate'], 2)} | {_f(r.get('auc_L1'))} | {_f(r.get('auc_L2'))} | {_f(r.get('auc_L3'))} | "
              f"{_f(r.get('auc_L4'))} | {_f(r.get('auc_L5'))} | {_f(r['ceiling_attribute_auc'])} |" for r in d["rows"]]
        L.append("")
    L += ["## Edge cost (one machine; the others idle)", ""]
    for m, c in s["cost"].items():
        L += [f"**{m}** (threads {c['threads']}, peak RSS {_f(c['peak_rss_mb'], 0)} MB, release for 1,000 users {_f(c['release_1000_users_s'], 2)} s):", "",
              "| Mode | Group | Users/call | p50 per user (ms) | p95 per user (ms) | call p50 (ms) |", "|---|---|---|---|---|---|"]
        L += [f"| {r['mode']} | {r['group_size']} | {r['users_per_call']} | {_f(r['per_user_p50_s'] * 1000, 1)} | {_f(r['per_user_p95_s'] * 1000, 1)} | {_f(r['call_p50_s'] * 1000, 1)} |" for r in c["rows"]]
        L.append("")
    if cov["infeasible_cells"] or cov["failed_cells"]:
        L += ["## Invalid cells", ""] + [f"- infeasible: {c}" for c in cov["infeasible_cells"]] + [f"- failed: {c}" for c in cov["failed_cells"]]
    L += ["", f"Analysis runtime: {s['runtime_min']} min."]
    return "\n".join(L) + "\n"
