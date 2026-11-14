"""Independent re-analysis of the CrossLeak v0.4 causal-defence study (own pooling of the per-target arrays, own
user-cluster bootstrap, three seeds). Does not import crossleak; reads the bundle zip directly.

    python docs/reanalysis/v4_reanalysis_v0.4.0.py <crossleak_v0.4.0_..._all-v4.zip>
"""

import io
import json
import sys
import zipfile

import numpy as np

MARGIN, B, SEEDS = 0.02, 5000, (1, 2, 3)


def load(bundle):
    cells = {}
    z = zipfile.ZipFile(bundle)
    for inner in sorted(n for n in z.namelist() if "/machines/" in n and n.endswith(".zip")):
        zi = zipfile.ZipFile(io.BytesIO(z.read(inner)))
        for n in zi.namelist():
            if "/runs/v4/util__" not in n or not n.endswith("/result.json"):
                continue
            r = json.loads(zi.read(n))
            assert r["status"] == "ok", n
            a = np.load(io.BytesIO(zi.read(n[: -len("result.json")] + "arrays.npz")), allow_pickle=False)
            cells.setdefault((r["dataset"], r["model"]), []).append((r["params"]["part"], {k: a[k] for k in a.files}))
    pooled = {}
    for key, parts in cells.items():
        parts.sort(key=lambda p: p[0])
        assert [p[0] for p in parts] == list(range(len(parts))), (key, [p[0] for p in parts])
        pooled[key] = {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
    return pooled


def gain(d, cfg, idx):
    return 1.0 - d["wql_" + cfg][idx].mean() / d["wql_iso"][idx].mean()


def boot_ci(d, fn, seed, q):
    rng = np.random.default_rng(seed)
    users = np.unique(d["user"])
    by = [np.flatnonzero(d["user"] == u) for u in users]
    vals = np.empty(B)
    for b in range(B):
        vals[b] = fn(np.concatenate([by[i] for i in rng.integers(0, len(users), len(users))]))
    return np.quantile(vals, q)


def main():
    pooled = load(sys.argv[1])
    print("== samples ==")
    for (ds, mod), d in sorted(pooled.items()):
        print(f"{ds:9} {mod:7} targets {len(d['user'])}  users {len(np.unique(d['user']))}  "
              f"distinct user-days {len(set(zip(d['user'].tolist(), d['day'].tolist())))}")
    print(f"\n== confirmatory pairs (c2base, TOST margin {MARGIN}, 90% CI, {B} resamples by user, seeds {SEEDS}) ==")
    pairs = [("eld", "d1p_16", "xl16"), ("eld", "d4_P8_eps1", "xl16"), ("eld", "d4_P8_eps1", "d1p_16"),
             ("lifesnaps", "d1p_4", "xl8"), ("lifesnaps", "d4p_P8_eps1", "xl8")]
    effects = [("eld", "d1p_16", "d1_16"), ("lifesnaps", "d1p_4", "d1_4"), ("lifesnaps", "d4p_P8_eps1", "d4_P8_eps1")]
    for title, rows in (("confirmatory", pairs), ("estimand: effect of causality", effects)):
        if title != "confirmatory":
            print(f"\n== {title} (c2base, 90% CI) ==")
        for ds, a, b in rows:
            d = pooled[(ds, "c2base")]
            allidx = np.arange(len(d["user"]))
            est = gain(d, a, allidx) - gain(d, b, allidx)
            cis = [boot_ci(d, lambda i: gain(d, a, i) - gain(d, b, i), s, (0.05, 0.95)) for s in SEEDS]
            ver = ["EQ" if (lo > -MARGIN and hi < MARGIN) else "NOT" for lo, hi in cis]
            print(f"{ds:9} {a:12} - {b:11}: {est:+.4f}  90% lo {min(c[0] for c in cis):+.4f}..{max(c[0] for c in cis):+.4f}"
                  f"  hi {min(c[1] for c in cis):+.4f}..{max(c[1] for c in cis):+.4f}  -> {ver}")
    print("\n== gains over isolated forecasting (95% CI, seed 1) ==")
    for (ds, mod), d in sorted(pooled.items()):
        allidx = np.arange(len(d["user"]))
        cfgs = [k[4:] for k in d if k.startswith("wql_") and k not in ("wql_iso", "wql_snaive")]
        out = []
        for c in cfgs:
            lo, hi = boot_ci(d, lambda i: gain(d, c, i), 1, (0.025, 0.975))
            out.append(f"{c} {gain(d, c, allidx):+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(f"{ds} {mod}: " + "; ".join(out))


if __name__ == "__main__":
    main()
