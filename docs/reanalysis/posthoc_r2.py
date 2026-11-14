"""Post hoc analyses (labelled; not pre-registered), run on the deposited v0.3 and v0.4 bundles after the final review.
Reads the per-target arrays of the utility cells (base model unless stated, one day of history). User-cluster
bootstrap, 5,000 resamples. Standard library + numpy; does not import crossleak.

  P1  Seasonal naive on the causal-study (v0.4) targets: gain(config) - gain(seasonal naive), 95 % CI.
  P2  D4 (P = 8, eps = 1) - cross-learning on ELD: (a) each study restricted to the cold-start users drawn in both;
      (b) both studies pooled (identical configuration, independent samples); (c) LifeSnaps v0.3 restricted to the
      origin days of v0.4. 90 % CI, margin 2 pp.
  P3  Margin sensitivity of the pre-registered equivalence tests: smallest symmetric margin each 90 % CI fits in.
  P4  Simultaneous intervals: Bonferroni over the five equivalence tests of each study (98 % CI).

    python docs/reanalysis/posthoc_r2.py --v3 <..._all-v3.zip> --v4 <..._all-v4.zip>
"""

import argparse
import io
import json
import zipfile

import numpy as np

B, SEED, MARGIN = 5000, 7, 0.02


def load(bundle, exp):
    cells = {}
    z = zipfile.ZipFile(bundle)
    for inner in sorted(n for n in z.namelist() if "/machines/" in n and n.endswith(".zip")):
        zi = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        for n in zi.namelist():
            if f"/runs/{exp}/util__" not in n or not n.endswith("/result.json"):
                continue
            r = json.loads(zi.read(n))
            if r["status"] != "ok" or r["params"]["ell"] != 24:
                continue
            a = np.load(io.BytesIO(zi.read(n[: -len("result.json")] + "arrays.npz")), allow_pickle=False)
            cells.setdefault((r["dataset"], r["model"]), []).append({k: a[k] for k in a.files})
    return {k: {c: np.concatenate([p[c] for p in v]) for c in v[0]} for k, v in cells.items()}


def summary(bundle, name):
    z = zipfile.ZipFile(bundle)
    return json.loads(z.read([n for n in z.namelist() if n.endswith(f"merged_report/{name}_summary.json")][0]))


def gain(d, c, i):
    return 1.0 - d["wql_" + c][i].mean() / d["wql_iso"][i].mean()


def boot(d, fn, q, cluster=None):
    rng = np.random.default_rng(SEED)
    cl = d["user"] if cluster is None else cluster
    groups = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    vals = np.array([fn(np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(B)])
    return fn(np.arange(len(cl))), np.quantile(vals, q[0]), np.quantile(vals, q[1])


def sub(d, mask):
    return {k: v[mask] for k, v in d.items()}


def fmt(t, s=100):
    return f"{s * t[0]:+.2f} [{s * t[1]:+.2f}, {s * t[2]:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3", required=True)
    ap.add_argument("--v4", required=True)
    a = ap.parse_args()
    v3, v4 = load(a.v3, "v3"), load(a.v4, "v4")
    s3, s4 = summary(a.v3, "v3"), summary(a.v4, "v4")

    print("== P1  seasonal naive on the v0.4 targets: gain(config) - gain(snaive), points, 95 % CI ==")
    for (ds, mod) in sorted(v4):
        d = v4[(ds, mod)]
        ok = np.isfinite(d["wql_snaive"])
        d = sub(d, ok)
        sn = boot(d, lambda i: gain(d, "snaive", i), (0.025, 0.975))
        print(f"{ds:9} {mod:7} n={ok.sum():3d} users={len(np.unique(d['user']))}: snaive gain vs isolated {fmt(sn)}")
        for c in [k[4:] for k in d if k.startswith("wql_") and k[4:] not in ("iso", "snaive", "full")]:
            r = boot(d, lambda i: gain(d, c, i) - gain(d, "snaive", i), (0.025, 0.975))
            print(f"    {c:12} - snaive {fmt(r)}")

    print("\n== P2  D4 (eps = 1) - cross-learning, 90 % CI (margin 2 pp) ==")
    e3, e4 = v3[("eld", "c2base")], v4[("eld", "c2base")]
    shared = np.intersect1d(np.unique(e3["user"]), np.unique(e4["user"]))
    for name, d in (("v0.3", e3), ("v0.4", e4)):
        m = np.isin(d["user"], shared)
        ds_ = sub(d, m)
        r = boot(ds_, lambda i: gain(ds_, "d4_P8_eps1", i) - gain(ds_, "xl16", i), (0.05, 0.95))
        print(f"(a) ELD {name}, {len(shared)} cold-start users drawn in both studies ({m.sum()} targets): {fmt(r)}")
    keys = ("user", "wql_iso", "wql_xl16", "wql_d4_P8_eps1")
    pooled = {k: np.concatenate([e3[k], e4[k]]) for k in keys}
    study = np.concatenate([np.zeros(len(e3["user"]), int), np.ones(len(e4["user"]), int)])
    clus = study * 100000 + pooled["user"]
    # pooled gain: each study's targets weighted equally by target count (600 each)
    r = boot(pooled, lambda i: gain(pooled, "d4_P8_eps1", i) - gain(pooled, "xl16", i), (0.05, 0.95), cluster=clus)
    print(f"(b) ELD both studies pooled (1,200 targets, clusters = study x user): {fmt(r)}")
    l3, l4 = v3[("lifesnaps", "c2base")], v4[("lifesnaps", "c2base")]
    m = l3["day"] >= l4["day"].min()
    ls_ = sub(l3, m)
    r = boot(ls_, lambda i: gain(ls_, "d4_P8_eps1", i) - gain(ls_, "xl8", i), (0.05, 0.95))
    print(f"(c) LifeSnaps v0.3 on origin days >= {l4['day'].min()} (the v0.4 window; {m.sum()} targets, "
          f"{len(np.unique(ls_['user']))} users): {fmt(r)}")

    print("\n== P3  smallest symmetric margin each pre-registered 90 % CI fits in (points) ==")
    for tag, s in (("v0.3 E1 ", s3), ("v0.4 E1c", s4)):
        for p in s["equivalence"]["pairs"]:
            mstar = 100 * max(abs(p["lo90"]), abs(p["hi90"]))
            print(f"{tag} {p['dataset']:9} {p['a']:12} - {p['b']:7}: {mstar:.2f}  (at 1.0: {'EQ' if mstar < 1 else 'not'}, "
                  f"at 1.5: {'EQ' if mstar < 1.5 else 'not'}, at 2.0: {'EQ' if mstar < 2 else 'not'})")

    print("\n== P4  Bonferroni over the five equivalence tests of each study: 98 % CI, margin 2 pp ==")
    for tag, data, s in (("v0.3 E1 ", v3, s3), ("v0.4 E1c", v4, s4)):
        for p in s["equivalence"]["pairs"]:
            d = data[(p["dataset"], "c2base")]
            r = boot(d, lambda i: gain(d, p["a"], i) - gain(d, p["b"], i), (0.01, 0.99))
            v = "EQUIVALENT" if (r[1] > -MARGIN and r[2] < MARGIN) else "NOT_EQUIVALENT"
            print(f"{tag} {p['dataset']:9} {p['a']:12} - {p['b']:7}: {fmt(r)}  {v}  (pre-registered: {p['verdict']})")


if __name__ == "__main__":
    main()
