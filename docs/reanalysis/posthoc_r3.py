"""Post hoc analyses requested in the manuscript review, round r3 (labelled; not pre-registered; approved by the author,
DECISIONS.md D-036). They use only the deposited bundles and the public datasets; no Chronos-2 forecast is run.

  PH-A  D4 (P = 8, eps = 1) - cross-learning (k = 16) on ELD, both studies pooled, resampling *unique users* (18 users
        appear in both studies): (i) pooled ratio of means; (ii) mean of the two study-specific differences (study
        effects). 90 % CI, exploratory.
  PH-B  A lightweight supervised baseline (crossleak.baselines): a global ridge forecaster on the last 24 hours
        (instance-scaled, with a missing-value mask and the weekday), trained on distinct windows of users the cold-start
        targets never belong to (auxiliary and public pools; ELD: 2013 only; LifeSnaps: windows before each origin,
        all days as a sensitivity), with per-step residual quantiles from a held-out quarter of the training users, which
        also selects the instance scaling (mean-absolute or mean/standard deviation). Repeated over five training
        seeds. Earlier runs (mean/std scaling only; then windows drawn with replacement and a held-out set that was not
        user-disjoint) are superseded (DECISIONS.md D-037). Scored by WQL on the stored v0.3 and v0.4
        cold-start targets and compared, paired, with isolated Chronos-2, cross-learning, public context, DP-Prototype
        and seasonal naive (95 % CI, user clusters). Alignment check: seasonal naive is recomputed from the data and
        must equal the stored value.
  PH-C  Point-forecast-only attacker: the attacks of the main study (pairs) and of the confirmatory study (residual
        attacker; pairs and k = 256 sybils) re-decoded from the median decile only (24 of the 216 features), next to
        the full features (which must reproduce the reported values).
  PH-D  Exact Gaussian ceilings of the DP audit: the optimal test between two values of one user's clipped
        representation under the release's Gaussian noise is mu-GDP with mu = 2 / sigma_s (worst case: both values in
        one cluster); its AUC is Phi(mu / sqrt 2). Compared with the (eps, delta) bound the paper reports.

    python docs/reanalysis/posthoc_r3.py --main <..._all-main.zip> --v3 <..._all-v3.zip> --v4 <..._all-v4.zip> \
        --work /tmp/ph_r3 [--n-boot 2000]

Requires the prepared datasets (``python -m crossleak data all``) for PH-B.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np

SEED = 7


# --------------------------------------------------------------------------- bundles


def extract_runs(bundle: str, exp: str, dest: Path) -> Path:
    """Extract the cell directories of one experiment (``runs/<exp>/``; smoke runs excluded) from a bundle's machine
    zips into ``dest``. Idempotent."""
    out = dest / exp
    if (out / ".done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(bundle)
    for inner in sorted(n for n in z.namelist() if "/machines/" in n and n.endswith(".zip")):
        zi = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        for n in zi.namelist():
            i = n.find(f"/runs/{exp}/")
            if i < 0 or n.endswith("/"):
                continue
            rel = n[i + len(f"/runs/{exp}/"):]
            if rel.startswith("/") or ".." in Path(rel).parts:
                raise ValueError(f"unsafe member path in {inner}: {n}")
            p = out / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(zi.read(n))
    (out / ".done").write_text("ok")
    return out


def util_arrays(root: Path, ell: int = 24) -> dict:
    cells: dict = {}
    for f in sorted(root.glob("util__*/result.json")):
        r = json.loads(f.read_text())
        if r["status"] != "ok" or r["params"]["ell"] != ell:
            continue
        a = np.load(f.parent / "arrays.npz", allow_pickle=False)
        cells.setdefault((r["dataset"], r["model"]), []).append({k: a[k] for k in a.files})
    return {k: {c: np.concatenate([p[c] for p in v]) for c in v[0]} for k, v in cells.items()}


# --------------------------------------------------------------------------- statistics


def gain(d: dict, c: str, i: np.ndarray) -> float:
    return 1.0 - np.nanmean(d["wql_" + c][i]) / np.nanmean(d["wql_iso"][i])


def boot(fn, cl: np.ndarray, q: tuple, n_boot: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    vals = np.array([fn(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(n_boot)])
    return float(fn(np.arange(len(cl)))), float(np.quantile(vals, q[0])), float(np.quantile(vals, q[1]))


def fmt(t, s=100.0, sign=True) -> str:
    f = "{:+.2f}" if sign else "{:.3f}"
    return f"{f.format(s * t[0])} [{f.format(s * t[1])}, {f.format(s * t[2])}]"


# --------------------------------------------------------------------------- PH-A


def ph_a(v3: dict, v4: dict, n_boot: int) -> dict:
    e3, e4 = v3[("eld", "c2base")], v4[("eld", "c2base")]
    keys = ("user", "wql_iso", "wql_xl16", "wql_d4_P8_eps1")
    P = {k: np.concatenate([e3[k], e4[k]]) for k in keys}
    study = np.concatenate([np.zeros(len(e3["user"]), int), np.ones(len(e4["user"]), int)])
    shared = np.intersect1d(np.unique(e3["user"]), np.unique(e4["user"]))

    def pooled(i):
        return gain(P, "d4_P8_eps1", i) - gain(P, "xl16", i)

    def study_effects(i):
        out = []
        for s in (0, 1):
            j = i[study[i] == s]
            out.append(gain(P, "d4_P8_eps1", j) - gain(P, "xl16", j))
        return float(np.mean(out))

    r1 = boot(pooled, P["user"], (0.05, 0.95), n_boot)
    r2 = boot(study_effects, P["user"], (0.05, 0.95), n_boot)
    old = boot(pooled, study * 100000 + P["user"], (0.05, 0.95), n_boot)
    res = {"n_targets": int(len(P["user"])), "n_unique_users": int(len(np.unique(P["user"]))), "n_shared_users": int(len(shared)),
           "pooled_unique_user": r1, "study_effects_unique_user": r2, "pooled_study_x_user_as_in_r2": old}
    print("== PH-A  ELD, D4 (eps = 1) - XL16, both studies, 90 % CI, points ==")
    print(f"targets {res['n_targets']}, unique users {res['n_unique_users']} ({res['n_shared_users']} in both studies)")
    print(f"(i)  pooled ratio of means, clusters = unique user:      {fmt(r1)}")
    print(f"(ii) mean of study-specific differences, unique user:     {fmt(r2)}")
    print(f"(r2) pooled, clusters = study x user (as posthoc_r2 P2b): {fmt(old)}")
    return res


# --------------------------------------------------------------------------- PH-B


def ph_b(roots: dict, n_boot: int, n_train: int, seeds: list[int]) -> dict:
    """Ridge baseline on the stored targets. Training: distinct (user, day) windows of auxiliary and public users; the
    held-out quarter (scaling choice, residual quantiles) is a quarter of those *users*. ELD trains on 2013 only;
    LifeSnaps (no training period) trains, for each origin, only on windows that end before it (primary) and, as a
    sensitivity, on all days. Repeated over training seeds (held-out users and, if capped, the window draw)."""
    from crossleak.baselines import fit_ridge, predict_ridge
    from crossleak.config import load_config, repo_root
    from crossleak.experiments.main import DS
    from crossleak.metrics import wql_item

    out: dict = {}
    print("\n== PH-B  global ridge baseline on the stored cold-start targets (one day of history), 95 % CI, points ==")
    for study, cfgname in (("v3", "v3.yaml"), ("v4", "v4.yaml")):
        cfg = load_config(repo_root() / "configs" / cfgname)
        U = util_arrays(roots[study])
        for ds_name in ("eld", "lifesnaps"):
            if (ds_name, "c2base") not in U:
                continue
            ds = DS(ds_name, cfg)
            d = U[(ds_name, "c2base")]
            n = len(d["user"])
            has_train = bool(cfg["datasets"][ds_name].get("train_days"))
            variants = [("primary", None if has_train else "past")] + ([] if has_train else [("all_days", None)])
            stored = d["wql_snaive"]
            w_sn = np.array([wql_item(ds.hor(int(u), int(dd)), np.repeat(ds.ctx(int(u), int(dd))[-24:][None], 9, axis=0))
                             for u, dd in zip(d["user"], d["day"])])
            both = np.isfinite(stored) & np.isfinite(w_sn)
            align = float(np.nanmax(np.abs(stored[both] - w_sn[both]))) if both.any() else float("nan")
            for vname, mode in variants:
                per_seed = []
                for seed in seeds:
                    t0 = time.time()
                    fits: dict = {}
                    w = np.full(n, np.nan)
                    for i, (u, dd) in enumerate(zip(d["user"], d["day"])):
                        key = int(dd) if mode == "past" else -1
                        if key not in fits:
                            try:
                                fits[key] = fit_ridge(ds, n_train, seed=seed, scaling="select", before_day=(int(dd) if mode == "past" else None))
                            except ValueError:
                                fits[key] = None
                        rm = fits[key]
                        if rm is None:
                            continue
                        w[i] = wql_item(ds.hor(int(u), int(dd)), predict_ridge(rm, ds.ctx(int(u), int(dd))[-24:], int(ds.weekday[int(dd)])))
                    first = next(f for f in fits.values() if f is not None)
                    per_seed.append({"seed": seed, "wql": w, "fit_s": time.time() - t0, "n_fits": len(fits),
                                     "scaling": sorted({f["scaling"] for f in fits.values() if f is not None}),
                                     "n_unique": first["n_unique"], "n_users": first["n_users"], "selection": first["selection"]})
                # primary statistics: the first seed; the spread over seeds as an estimand
                row = {"variant": vname, "training": ("train split (2013)" if has_train else ("windows before each origin" if mode == "past" else "all days")),
                       "snaive_alignment_max_abs": align, "seeds": seeds, "fits": [{k: v for k, v in r.items() if k != "wql"} for r in per_seed]}
                gains_by_seed = []
                for r in per_seed:
                    dd_ = dict(d)
                    dd_["wql_ridge"] = r["wql"]
                    ok = np.isfinite(dd_["wql_ridge"]) & np.isfinite(dd_["wql_snaive"])
                    gains_by_seed.append(gain({k: v[ok] for k, v in dd_.items()}, "ridge", np.arange(ok.sum())))
                row["ridge_gain_by_seed"] = gains_by_seed
                # cross-learning minus ridge in every seed (95 % CI by user): is the comparison robust to the seed?
                xl_by_seed = {}
                for r in per_seed:
                    dd_ = dict(d)
                    dd_["wql_ridge"] = r["wql"]
                    ok = np.isfinite(dd_["wql_ridge"]) & np.isfinite(dd_["wql_snaive"])
                    dd_ = {k: v[ok] for k, v in dd_.items()}
                    for c in [c[4:] for c in dd_ if c.startswith("wql_xl")]:
                        xl_by_seed.setdefault(c, []).append(boot(lambda i: gain(dd_, c, i) - gain(dd_, "ridge", i), dd_["user"], (0.025, 0.975), n_boot))
                row["xl_minus_ridge_by_seed"] = xl_by_seed
                dd_ = dict(d)
                dd_["wql_ridge"] = per_seed[0]["wql"]
                ok = np.isfinite(dd_["wql_ridge"]) & np.isfinite(dd_["wql_snaive"])
                dd_ = {k: v[ok] for k, v in dd_.items()}
                cl = dd_["user"]
                configs = [c[4:] for c in dd_ if c.startswith("wql_") and c[4:] not in ("iso", "full", "ridge", "snaive")]
                row.update({"n_targets": int(ok.sum()), "n_users": int(len(np.unique(cl))), "gain_vs_iso": {}, "minus_ridge": {}})
                for c in ["ridge", "snaive"] + configs:
                    row["gain_vs_iso"][c] = boot(lambda i: gain(dd_, c, i), cl, (0.025, 0.975), n_boot)
                for c in ["snaive"] + configs:
                    row["minus_ridge"][c] = boot(lambda i: gain(dd_, c, i) - gain(dd_, "ridge", i), cl, (0.025, 0.975), n_boot)
                out[f"{study}_{ds_name}" + ("" if vname == "primary" else f"_{vname}")] = row
                f0 = per_seed[0]
                print(f"-- {study} {ds_name} [{row['training']}]: {row['n_targets']} targets, {row['n_users']} users; "
                      f"{f0['n_unique']} distinct training windows of {f0['n_users']} users, {f0['n_fits']} fit(s), scaling {f0['scaling']}; "
                      f"ridge gain over seeds {seeds}: " + ", ".join(f"{100 * g:+.2f}" for g in gains_by_seed)
                      + f"; snaive alignment max |diff| {align:.2e}")
                for c, r in row["gain_vs_iso"].items():
                    extra = f"   minus ridge {fmt(row['minus_ridge'][c])}" if c in row["minus_ridge"] else ""
                    print(f"    gain vs isolated Chronos-2: {c:12} {fmt(r)}{extra}")
                for c, rs in xl_by_seed.items():
                    print(f"    {c} - ridge by seed: " + "; ".join(fmt(x) for x in rs))
    return out


# --------------------------------------------------------------------------- PH-C


MEDIAN_COLS = slice(4 * 24, 5 * 24)  # decile-major (9, 24) -> the 0.5 decile


def ph_c(roots: dict, n_boot: int) -> dict:
    from crossleak.analysis import main_report as MR
    from crossleak.analysis import v3_report as V3

    orig_arrays = MR.arrays

    def point_only(r):
        a = orig_arrays(r)
        if "fxl" in a:
            a["fxl"] = a["fxl"][:, MEDIAN_COLS]
            a["fiso"] = a["fiso"][:, MEDIAN_COLS]
        return a

    out: dict = {}
    print("\n== PH-C  point-forecast-only attacker (median decile) vs all nine deciles ==")
    for study, exp in (("main", "main"), ("v3", "v3")):
        results, _ = MR.load_results([roots[study]])
        keep = {}
        for k, r in results.items():
            p = r.get("params", {})
            if r["exp"] == "ref" or (r["exp"] == "attack" and p.get("probe") == "short" and (
                    (p.get("fam") == "single" and p.get("k") == 2 and r["model"] == "c2base")
                    or (study == "v3" and p.get("fam") == "sybil" and p.get("k") == 256 and r["dataset"] == "eld"))):
                keep[k] = r
        for label, fn in (("all deciles", orig_arrays), ("median only", point_only)):
            MR.arrays = fn
            try:
                if study == "main":
                    res = MR.analyse_attacks(keep, "linear", n_boot, 1)
                else:
                    res = V3._residual_attacker(keep, n_boot, 1)
            finally:
                MR.arrays = orig_arrays
            for key, r in sorted(res.items()):
                tag = f"{study} {r.get('dataset')} {r.get('model')} {r.get('fam')} k={r.get('k')}"
                auc = r.get("auc_L1", {})
                dauc = r.get("delta_auc_L1", {})
                dr = r.get("delta_r1", {})
                g = lambda x: (x.get("est"), x.get("lo"), x.get("hi")) if isinstance(x, dict) else None  # noqa: E731
                out.setdefault(tag, {})[label] = {"auc_L1": g(auc), "auc_L1_ctrl": r.get("auc_L1_ctrl"), "delta_auc_L1": g(dauc),
                                                  "delta_r1": g(dr), "n_train": r.get("n_train"), "n_test": r.get("n_test")}
                print(f"{tag:42} {label:11}: AUC L1 {fmt(g(auc), 1, False) if g(auc) else 'n/a'}  "
                      f"dAUC {fmt(g(dauc), 1, False) if g(dauc) else 'n/a'}  dR1 {fmt(g(dr), 1, False) if g(dr) else 'n/a'}")
    return out


# --------------------------------------------------------------------------- PH-D


def ph_d() -> dict:
    from crossleak.defenses import dp_prototype as dp

    from crossleak.config import load_config, repo_root

    c3, c4 = (load_config(repo_root() / "configs" / f) for f in ("v3.yaml", "v4.yaml"))
    assert c3["delta"] == c4["delta"] and c3["d4"]["sum_share"] == c4["d4"]["sum_share"]
    delta, share = float(c4["delta"]), float(c4["d4"]["sum_share"])
    # worst case: both values of the user in one cluster (mu^2 = 4 / sigma_s^2) dominates two clusters
    # (2 / sigma_s^2 + 2 / sigma_n^2) iff sigma_n >= sigma_s, i.e. sum_share >= 0.5
    assert share >= 0.5
    print("\n== PH-D  DP audit ceilings: (eps, delta) bound vs exact Gaussian (mu-GDP) ==")
    out = {}
    for eps in (0.5, 1.0, 4.0):
        sc = dp.noise_scales(eps, delta, share)
        mu = 2.0 / sc.sigma_sum
        exact = 0.5 * (1 + math.erf(mu / math.sqrt(2) / math.sqrt(2)))
        bound = dp.attribute_ceiling(eps, delta)
        out[str(eps)] = {"sigma_sum": sc.sigma_sum, "mu": mu, "auc_exact_gaussian": exact, "auc_eps_delta_bound": bound}
        print(f"eps={eps:<3}: sigma_s={sc.sigma_sum:.3f}  mu={mu:.3f}  exact Gaussian AUC ceiling {exact:.3f}  "
              f"(eps,delta) bound {bound:.3f}")
    return out


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v4", required=True)
    ap.add_argument("--work", default="/tmp/ph_r3")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-train", type=int, default=10 ** 6, help="cap on distinct training windows (default: all)")
    ap.add_argument("--seeds", default="11,12,13,14,15", help="PH-B training seeds (the first is primary)")
    ap.add_argument("--only", default="A,B,C,D")
    ap.add_argument("--out", default=None, help="JSON output (default: <work>/posthoc_r3.json)")
    a = ap.parse_args()
    work = Path(a.work)
    t0 = time.time()
    roots = {"main": extract_runs(a.main, "main", work), "v3": extract_runs(a.v3, "v3", work), "v4": extract_runs(a.v4, "v4", work)}
    only = set(a.only.split(","))
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    res: dict = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_boot": a.n_boot,
                 "bundles": {k: {"name": Path(v).name, "sha256": sha(v)} for k, v in (("main", a.main), ("v3", a.v3), ("v4", a.v4))}}
    if "A" in only:
        res["PH_A"] = ph_a(util_arrays(roots["v3"]), util_arrays(roots["v4"]), a.n_boot)
    if "D" in only:
        res["PH_D"] = ph_d()
    if "B" in only:
        res["PH_B"] = ph_b(roots, a.n_boot, a.n_train, [int(x) for x in a.seeds.split(",")])
    if "C" in only:
        res["PH_C"] = ph_c(roots, a.n_boot)
    res["runtime_min"] = round((time.time() - t0) / 60, 1)
    out = Path(a.out) if a.out else work / "posthoc_r3.json"
    out.write_text(json.dumps(res, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    print(f"\nwritten {out} ({res['runtime_min']} min)")


if __name__ == "__main__":
    main()
