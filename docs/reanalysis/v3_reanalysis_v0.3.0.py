"""Independent re-analysis of the CrossLeak v0.3 decisive results (own bootstrap, own labels from the raw panels, own
decoders with 5-fold CV). Does not import crossleak.attack, crossleak.analysis or crossleak.decode.

    python docs/reanalysis/v3_reanalysis_v0.3.0.py --runs 'unpacked/*/*/runs/v3' --data ~/crossleak-data --src src
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--src", required=True)
args = ap.parse_args()
sys.path.insert(0, args.src)
os.environ["CROSSLEAK_DATA"] = args.data
from crossleak.data.panel import load_panel  # noqa: E402  (data access only)

ROOTS = glob.glob(args.runs)


def cells(prefix):
    out = []
    for r in ROOTS:
        for d in glob.glob(os.path.join(r, prefix + "*")):
            res = json.load(open(os.path.join(d, "result.json")))
            out.append((res, dict(np.load(os.path.join(d, "arrays.npz")))))
    return out


def boot(stat, clusters, n, seed, q=(0.05, 0.95)):
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
    return float(stat(np.arange(len(clusters)))), float(np.quantile(vals, q[0])), float(np.quantile(vals, q[1]))


def util(ds, model, ell):
    parts = [(r, a) for r, a in cells(f"util__{ds}__{model}__") if r["params"]["ell"] == ell]
    W = {k[4:]: np.concatenate([a[k] for _, a in parts]) for k in parts[0][1] if k.startswith("wql_")}
    return W, np.concatenate([a["user"] for _, a in parts])


print("== equivalence (own bootstrap by user, 90% CI, 5,000 resamples, three seeds) ==")
specs = [("eld", 24, "d1_16", "xl16"), ("eld", 24, "d4_P8_eps1", "xl16"), ("eld", 24, "d4_P8_eps1", "d1_16"),
         ("lifesnaps", 24, "d1_4", "xl8"), ("lifesnaps", 24, "d4_P8_eps1", "xl8")]
for ds, ell, a, b in specs:
    W, U = util(ds, "c2base", ell)
    ok = np.isfinite(W["iso"]) & np.isfinite(W[a]) & np.isfinite(W[b])
    iso, wa, wb, uu = W["iso"][ok], W[a][ok], W[b][ok], U[ok]
    res = [boot(lambda i: (wb[i].mean() - wa[i].mean()) / iso[i].mean(), uu, 5000, s) for s in (11, 12, 13)]
    his = [r[2] for r in res]
    los = [r[1] for r in res]
    verdict = ["EQ" if (lo > -0.02 and hi < 0.02) else "NOT" for lo, hi in zip(los, his)]
    print(f"{ds:9} {a:11} - {b:6}: {res[0][0]:+.4f}  90% lo {min(los):+.4f}..{max(los):+.4f}  hi {min(his):+.4f}..{max(his):+.4f}  -> {verdict}  n={len(iso)} users={len(np.unique(uu))}")
for ds, ell, c in (("eld", 72, "xl16"), ("eld", 72, "xl4")):
    W, U = util(ds, "c2base", ell)
    ok = np.isfinite(W["iso"]) & np.isfinite(W[c])
    iso, w, uu = W["iso"][ok], W[c][ok], U[ok]
    res = [boot(lambda i: 1 - w[i].mean() / iso[i].mean(), uu, 5000, s) for s in (11, 12, 13)]
    print(f"{ds} ell={ell} {c:5} gain {res[0][0]:+.4f} 90% [{min(r[1] for r in res):+.4f}, {max(r[2] for r in res):+.4f}]")

print("\n== seasonal-naive baseline (repeat the last 24 h), paired, 95% CI ==")
for ds, model, ell in (("eld", "c2base", 24), ("eld", "c2small", 24), ("lifesnaps", "c2base", 24), ("eld", "c2base", 72)):
    W, U = util(ds, model, ell)
    iso = W["iso"]
    for c in [x for x in ("xl16", "xl8", "d1_16", "d4_P8_eps1", "d4_P8_eps4", "full") if x in W]:
        ok = np.isfinite(iso) & np.isfinite(W[c]) & np.isfinite(W["snaive"])
        sn, wc, i0, uu = W["snaive"][ok], W[c][ok], iso[ok], U[ok]
        r = boot(lambda i: (sn[i].mean() - wc[i].mean()) / i0[i].mean(), uu, 2000, 21, (0.025, 0.975))
        print(f"{ds:9} {model} ell={ell} {c:11}: gain(config) - gain(snaive) {r[0]:+.4f} [{r[1]:+.4f}, {r[2]:+.4f}]")

# ---------------------------------------------------------------- residual attacker with own labels
print("\n== stronger sybil attacker: own labels from raw data, own decoders (95% CI by victim) ==")
panels = {ds: load_panel(ds) for ds in ("eld", "lifesnaps")}
ref = {}
for ds in ("eld", "lifesnaps"):
    ref[ds] = cells(f"ref__{ds}__none")[0][1]


def day_z(ds, u, oh):
    x = panels[ds].values[u, oh - 24 : oh].astype(float)
    m = np.isfinite(x)
    if m.sum() < 20:
        return None
    t = np.arange(24)
    x = np.interp(t, t[m], x[m])
    s = x.std()
    return (x - x.mean()) / s if s > 0 else None


cache = {}


def day_median(ds, d, oh, users):
    key = (ds, d, len(users), int(users[0]))
    if key not in cache:
        ev = [z[18:24].mean() for z in (day_z(ds, int(u), oh) for u in users) if z is not None]
        cache[key] = np.median(ev) if ev else np.nan
    return cache[key]


def origin_hour(ds, d):
    t0 = np.datetime64(str(panels[ds].index[0]))
    return int((np.datetime64(str(ref[ds]["days"][d])) - t0) / np.timedelta64(1, "h"))


def own_logit(Xtr, y, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    best, bc = -1, None
    for C in (1e-4, 1e-3, 1e-2, 1e-1, 1, 10):
        sc = [roc_auc_score(y[va], LogisticRegression(C=C, max_iter=3000).fit(A[tr], y[tr]).decision_function(A[va]))
              for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(A, y)]
        if np.mean(sc) > best:
            best, bc = np.mean(sc), C
    return LogisticRegression(C=bc, max_iter=3000).fit(A, y).decision_function(B)


def own_ridge(Xtr, Y, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    best, ba = 1e18, None
    for al in 10.0 ** np.arange(-1, 7):
        e = sum(((Ridge(alpha=al).fit(A[tr], Y[tr]).predict(A[va]) - Y[va]) ** 2).sum() for tr, va in KFold(5, shuffle=True, random_state=0).split(A))
        if e < best:
            best, ba = e, al
    return Ridge(alpha=ba).fit(A, Y).predict(B)


def corr_rows(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)


def family(ds, model, fam, k):
    tr, te = [], []
    for res, a in cells(f"attack__{ds}__{model}__"):
        p = res["params"]
        if (p["fam"], p["k"]) == (fam, k):
            (tr if p["role"] == "train" else te).append(a)
    cat = lambda P, key: np.concatenate([x[key] for x in P])  # noqa: E731
    return {key: cat(tr, key) for key in tr[0]}, {key: cat(te, key) for key in te[0]}


for ds, model, fam, k in (("eld", "c2small", "sybil", 256), ("eld", "c2small", "sybil", 64), ("lifesnaps", "c2small", "sybil", 256),
                          ("eld", "c2base", "sybil", 16), ("eld", "c2base", "single", 2)):
    tr, te = family(ds, model, fam, k)
    pools = ref[ds]
    att = np.concatenate([pools["pools_aux"], pools["pools_public"]])
    allu = np.arange(panels[ds].n_users)

    def labels(T, users):
        y, Y = [], []
        for v, d in zip(T["victim"], T["day"]):
            oh = origin_hour(ds, int(d))
            z = day_z(ds, int(v), oh)
            med = day_median(ds, int(d), oh, users)
            y.append(-1 if (z is None or not np.isfinite(med)) else int(z[18:24].mean() > med))
            Y.append(np.full(24, np.nan) if z is None else z)
        return np.array(y), np.array(Y)

    ytr, Ytr = labels(tr, att)
    yte, Yte = labels(te, allu)
    f32 = lambda a: np.asarray(a, dtype=float)  # noqa: E731
    Rtr = np.hstack([f32(tr["fxl"]) - f32(tr["fiso"]), f32(tr["cal"])])
    Rte = np.hstack([f32(te["fxl"]) - f32(te["fiso"]), f32(te["cal"])])
    Ctr = np.hstack([f32(tr["fiso"]), f32(tr["cal"])])
    Cte = np.hstack([f32(te["fiso"]), f32(te["cal"])])
    mtr, mte = ytr >= 0, yte >= 0
    sx = own_logit(Rtr[mtr], ytr[mtr], Rte[mte])
    sc = own_logit(Ctr[mtr], ytr[mtr], Cte[mte])
    yy, vv = yte[mte], te["victim"][mte]
    a_ = boot(lambda i: roc_auc_score(yy[i], sx[i]), vv, 2000, 4, (0.025, 0.975))
    d_ = boot(lambda i: roc_auc_score(yy[i], sx[i]) - roc_auc_score(yy[i], sc[i]), vv, 2000, 5, (0.025, 0.975))
    okp = np.isfinite(Ytr).all(1)
    yh, yc = own_ridge(Rtr[okp], Ytr[okp], Rte), own_ridge(Ctr[okp], Ytr[okp], Cte)
    means = np.array([np.mean([z for z in (day_z(ds, int(u), origin_hour(ds, int(d))) for u in allu) if z is not None], axis=0) for d in te["day"]])
    okt = np.isfinite(Yte).all(1)
    rx, rc = corr_rows(yh[okt] - means[okt], Yte[okt] - means[okt]), corr_rows(yc[okt] - means[okt], Yte[okt] - means[okt])
    dr = boot(lambda i: float(np.mean(rx[i] - rc[i])), te["victim"][okt], 2000, 6, (0.025, 0.975))
    agree = np.mean(yte[mte] == (te["Y"][mte][:, 18:24].mean(1) > ref[ds]["ref_even_med"][te["day"][mte]]).astype(int))
    print(f"{ds:9} {model} {fam:6} k={k:3}: residual AUC L1 {a_[0]:.3f} [{a_[1]:.3f}, {a_[2]:.3f}] ctrl {roc_auc_score(yy, sc):.3f} "
          f"dAUC {d_[0]:+.3f} [{d_[1]:+.3f}, {d_[2]:+.3f}] | dr1 {dr[0]:+.3f} [{dr[1]:+.3f}, {dr[2]:+.3f}] | label agreement {agree:.3f} "
          f"| n_tr {mtr.sum()} n_te {mte.sum()}", flush=True)
