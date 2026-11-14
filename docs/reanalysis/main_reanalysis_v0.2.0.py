"""Independent re-analysis of the CrossLeak main-study decisive cells (own labels from the raw panels, own decoders with
5-fold CV, own bootstrap). It does not import crossleak.attack, crossleak.analysis or crossleak.decode.

    python docs/reanalysis/main_reanalysis_v0.2.0.py --runs 'unpacked/*/*/runs/main' > main_reanalysis_v0.2.0.out
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--runs", required=True, help="glob of run directories, e.g. 'unpacked/*/*/runs/main' (quote it)")
ap.add_argument("--data", default=os.environ.get("CROSSLEAK_DATA", os.path.expanduser("~/crossleak-data")))
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
args = ap.parse_args()
sys.path.insert(0, args.src)
os.environ["CROSSLEAK_DATA"] = args.data
from crossleak.data.panel import load_panel  # noqa: E402  (data access only)

ROOTS = glob.glob(args.runs)
B = 1000
rng0 = np.random.default_rng(12345)


def cells(prefix):
    out = []
    for r in ROOTS:
        for d in glob.glob(os.path.join(r, prefix + "*")):
            res = json.load(open(os.path.join(d, "result.json")))
            out.append((res, dict(np.load(os.path.join(d, "arrays.npz")))))
    return out


def boot(stat, clusters, n=B, seed=1):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(u))]
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        try:
            v = stat(idx)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    return float(stat(np.arange(len(clusters)))), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


# ---------------------------------------------------------------- utility
print("== utility (c2base) ==")
for ds in ("eld", "lifesnaps"):
    for ell in (24, 72):
        parts = [(r, a) for r, a in cells(f"util__{ds}__c2base__") if r["params"]["ell"] == ell]
        W = {k[4:]: np.concatenate([a[k] for _, a in parts]) for k in parts[0][1] if k.startswith("wql_")}
        users = np.concatenate([a["user"] for _, a in parts])
        iso = W["iso"]
        show = [c for c in W if c.startswith("xl")] + [c for c in ("d1_16", "d1_4", "d4_P8_eps1", "d4_P8_eps4", "d4_P4_eps1") if c in W]
        for c in show:
            est = boot(lambda i: 1 - W[c][i].mean() / iso[i].mean(), users, seed=2)
            print(f"{ds} ell={ell} {c:12} gain {est[0]:+.4f} [{est[1]:+.4f}, {est[2]:+.4f}]  n={len(iso)} users={len(np.unique(users))}")
        if ell == 24:
            ref = "xl16" if ds == "eld" else "xl8"
            d1 = "d1_16" if ds == "eld" else "d1_4"
            for a_, b_ in ((f"d4_P8_eps1", d1), (f"d4_P8_eps1", ref), (d1, ref), ("d4_P8_eps4", d1)):
                est = boot(lambda i: (W[b_][i].mean() - W[a_][i].mean()) / iso[i].mean(), users, seed=3)
                print(f"   paired gain difference {a_} - {b_}: {est[0]:+.4f} [{est[1]:+.4f}, {est[2]:+.4f}]")

# ---------------------------------------------------------------- labels from raw data
panels = {ds: load_panel(ds) for ds in ("eld", "lifesnaps")}
H = 24


def day_z(ds, u, origin_hour):
    x = panels[ds].values[u, origin_hour - 24 : origin_hour].astype(float)
    m = np.isfinite(x)
    if m.sum() < 20:
        return None
    t = np.arange(24)
    x = np.interp(t, t[m], x[m])
    s = x.std()
    return (x - x.mean()) / s if s > 0 else None


ref_cache = {}


def day_median(ds, d_index, origin_hour, users):
    key = (ds, d_index, tuple(users[:3]), len(users))
    if key not in ref_cache:
        ev = []
        for u in users:
            z = day_z(ds, int(u), origin_hour)
            if z is not None:
                ev.append(z[18:24].mean())
        ref_cache[key] = np.median(ev) if ev else np.nan
    return ref_cache[key]


def own_decoder(Xtr, ytr, Xte, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    best, bestC = -1, None
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1, 10):
        sc = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, ytr):
            m = LogisticRegression(C=C, max_iter=3000).fit(A[tr], ytr[tr])
            sc.append(roc_auc_score(ytr[va], m.decision_function(A[va])))
        if np.mean(sc) > best:
            best, bestC = np.mean(sc), C
    return LogisticRegression(C=bestC, max_iter=3000).fit(A, ytr).decision_function(Bm), bestC


def own_ridge(Xtr, Ytr, Xte, seed=0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, Bm = (Xtr - mu) / sd, (Xte - mu) / sd
    best, bestA = 1e18, None
    for al in 10.0 ** np.arange(-1, 7):
        err = 0
        for tr, va in KFold(5, shuffle=True, random_state=seed).split(A):
            m = Ridge(alpha=al).fit(A[tr], Ytr[tr])
            err += ((m.predict(A[va]) - Ytr[va]) ** 2).sum()
        if err < best:
            best, bestA = err, al
    return Ridge(alpha=bestA).fit(A, Ytr).predict(Bm)


def corr_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


ref_any = {}
for ds in ("eld", "lifesnaps"):
    r = cells(f"ref__{ds}__none")[0][1]
    ref_any[ds] = r


def family(ds, model, fam, filler, k, probe="short"):
    tr, te = [], []
    for res, a in cells(f"attack__{ds}__{model}__"):
        p = res["params"]
        if (p["fam"], p["filler"], p["k"], p["probe"]) == (fam, filler, k, probe):
            (tr if p["role"] == "train" else te).append(a)
    cat = lambda parts, key: np.concatenate([x[key] for x in parts])  # noqa: E731
    return {key: cat(tr, key) for key in tr[0]}, {key: cat(te, key) for key in te[0]}


def origin_hour_of(ds, d):
    # origins are midnights from the 15th day on: recompute from the validity origins of the ref cell's day list
    days = ref_any[ds]["days"]
    t0 = np.datetime64(str(panels[ds].index[0]))
    return int((np.datetime64(str(days[d])) - t0) / np.timedelta64(1, "h"))


print("\n== attacks (own labels from raw data, own decoders) ==")
for ds, model, fam, filler, k in [("eld", "c2base", "single", "none", 2), ("lifesnaps", "c2base", "single", "none", 2),
                                   ("eld", "c2small", "single", "none", 2), ("eld", "c2base", "sybil", "copy", 16),
                                   ("lifesnaps", "c2base", "sybil", "copy", 16), ("eld", "c2small", "sybil", "copy", 64),
                                   ("eld", "c2small", "sybil", "copy", 256), ("lifesnaps", "c2small", "sybil", "copy", 256)]:
    tr, te = family(ds, model, fam, filler, k)
    pools = ref_any[ds]
    att_users = np.concatenate([pools["pools_aux"], pools["pools_public"]])
    all_users = np.arange(panels[ds].n_users)

    def labels(T, users):
        y, Yz = [], []
        for v, d in zip(T["victim"], T["day"]):
            oh = origin_hour_of(ds, int(d))
            z = day_z(ds, int(v), oh)
            med = day_median(ds, int(d), oh, users)
            y.append(-1 if (z is None or not np.isfinite(med)) else int(z[18:24].mean() > med))
            Yz.append(np.full(24, np.nan) if z is None else z)
        return np.array(y), np.array(Yz)

    ytr, Ytr = labels(tr, att_users)
    yte, Yte = labels(te, all_users)
    Xtr = np.hstack([tr["fxl"], tr["fiso"], tr["cal"]]).astype(float)
    Xte = np.hstack([te["fxl"], te["fiso"], te["cal"]]).astype(float)
    Ctr = np.hstack([tr["fiso"], tr["cal"]]).astype(float)
    Cte = np.hstack([te["fiso"], te["cal"]]).astype(float)
    mtr, mte = ytr >= 0, yte >= 0
    sx, cx = own_decoder(Xtr[mtr], ytr[mtr], Xte[mte])
    sc, cc = own_decoder(Ctr[mtr], ytr[mtr], Cte[mte])
    vv, yy = te["victim"][mte], yte[mte]
    a = boot(lambda i: roc_auc_score(yy[i], sx[i]), vv, seed=4)
    dlt = boot(lambda i: roc_auc_score(yy[i], sx[i]) - roc_auc_score(yy[i], sc[i]), vv, seed=5)
    # profile: within-day anomaly with my own population mean over all users that day
    okp = np.isfinite(Ytr).all(1)
    yh = own_ridge(Xtr[okp], Ytr[okp], Xte)
    yc = own_ridge(Ctr[okp], Ytr[okp], Cte)
    means = []
    for d in te["day"]:
        oh = origin_hour_of(ds, int(d))
        zs = [day_z(ds, int(u), oh) for u in all_users]
        zs = [z for z in zs if z is not None]
        means.append(np.mean(zs, axis=0))
    means = np.array(means)
    okt = np.isfinite(Yte).all(1)
    rx = corr_rows(yh[okt] - means[okt], Yte[okt] - means[okt])
    rc = corr_rows(yc[okt] - means[okt], Yte[okt] - means[okt])
    dr = boot(lambda i: float(np.mean(rx[i] - rc[i])), te["victim"][okt], seed=6)
    agree = np.mean((yte[mte] == (te["Y"][mte][:, 18:24].mean(1) > ref_any[ds]["ref_even_med"][te["day"][mte]]).astype(int)))
    print(f"{ds:9} {model} {fam:6} {filler:5} k={k:3}: AUC L1 {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] ctrl {roc_auc_score(yy, sc):.3f} "
          f"dAUC {dlt[0]:+.3f} [{dlt[1]:+.3f}, {dlt[2]:+.3f}] | dr1 {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}] | "
          f"label agreement with pipeline {agree:.3f} | C {cx}/{cc} n_tr {mtr.sum()} n_te {mte.sum()}", flush=True)

# ---------------------------------------------------------------- channel ratios
print("\n== channel ratios (own medians, independent bootstrap) ==")
def chan(ds, model, fam, filler, k):
    xs, vs = [], []
    for res, a in cells(f"channel__{ds}__{model}__"):
        p = res["params"]
        if (p["fam"], p["filler"], p["k"]) == (fam, filler, k):
            xs.append(a["i_swap"]); vs.append(a["victim"])
    return np.concatenate(xs), np.concatenate(vs)
for ds, kmax in (("eld", 256), ("lifesnaps", 8)):
    for model in ("c2small", "c2base"):
        x2, v2 = chan(ds, model, "single", "none", 2)
        for fam, filler, k in (("single", "none", kmax), ("sybil", "copy", 256)):
            xk, vk = chan(ds, model, fam, filler, k)
            rng = np.random.default_rng(9)
            vals = [np.median(rng.choice(xk, len(xk))) / np.median(rng.choice(x2, len(x2))) for _ in range(B)]
            print(f"{ds} {model} {fam} k={k}: ratio {np.median(xk) / np.median(x2):.3f} [{np.quantile(vals, .025):.3f}, {np.quantile(vals, .975):.3f}] (pairs {len(xk)})")
