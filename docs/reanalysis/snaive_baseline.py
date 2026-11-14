"""Post hoc (labelled): the seasonal-naive baseline (repeat the last 24 h as a point forecast; its WQL equals its WAPE)
against the cold-start configurations, paired on the same targets, 95% CI by user (2,000 resamples). Reads the
`wql_snaive` arrays that every utility cell already stored.

    python docs/reanalysis/snaive_baseline.py --runs 'unpacked/*/*/runs/main' --label v0.2
    python docs/reanalysis/snaive_baseline.py --runs 'unpacked/*/*/runs/v3' --label v0.3
"""

import argparse
import glob
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
roots = glob.glob(a.runs)


def util(ds, model, ell):
    parts = []
    for r in roots:
        for d in glob.glob(os.path.join(r, f"util__{ds}__{model}__*")):
            res = json.load(open(os.path.join(d, "result.json")))
            if res["params"]["ell"] == ell:
                parts.append(dict(np.load(os.path.join(d, "arrays.npz"))))
    if not parts:
        return None, None
    W = {k[4:]: np.concatenate([p[k] for p in parts]) for k in parts[0] if k.startswith("wql_")}
    return W, np.concatenate([p["user"] for p in parts])


def boot(stat, cl, n=2000, seed=21):
    rng = np.random.default_rng(seed)
    u, inv = np.unique(cl, return_inverse=True)
    g = [np.where(inv == i)[0] for i in range(len(u))]
    v = [stat(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(n)]
    return stat(np.arange(len(cl))), np.quantile(v, 0.025), np.quantile(v, 0.975)


for ds, model, ell in (("eld", "c2base", 24), ("eld", "c2small", 24), ("eld", "c2base", 72), ("lifesnaps", "c2base", 24), ("lifesnaps", "c2small", 24)):
    W, U = util(ds, model, ell)
    if W is None:
        continue
    ok = np.isfinite(W["iso"]) & np.isfinite(W["snaive"])
    for c in W:
        ok &= np.isfinite(W[c])
    iso, sn, U = W["iso"][ok], W["snaive"][ok], U[ok]
    g = boot(lambda i: 1 - sn[i].mean() / iso[i].mean(), U)
    print(f"{a.label} {ds:9} {model} ell={ell}: snaive gain vs isolated {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]  (n={len(iso)})")
    for c in [x for x in ("xl4", "xl8", "xl16", "d1_4", "d1_16", "d4_P8_eps1", "d4_P8_eps4") if x in W]:
        wc = W[c][ok]
        d = boot(lambda i: (sn[i].mean() - wc[i].mean()) / iso[i].mean(), U)
        print(f"{a.label} {ds:9} {model} ell={ell}:   gain({c}) - gain(snaive) {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]")
