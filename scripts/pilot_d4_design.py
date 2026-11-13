#!/usr/bin/env python3
"""Design pilot for D4 (DP-Prototype), run BEFORE the main pre-registration freeze.

It uses only ELD 2013 (the attacker-training period, never used to evaluate the main claims) and chooses the
prototype representation and P by the cold-start gain at a fixed epsilon. Output: one JSON file.

    python scripts/pilot_d4_design.py --model c2small --n 80 --out docs/pilots/d4_design_c2small.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import yaml

from crossleak.config import stable_seed
from crossleak.data.panel import compute_validity, day_mask, load_panel, make_pools
from crossleak.defenses import dp_prototype as dp
from crossleak.metrics import wql_item
from crossleak.model import Forecaster, pad_left


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="c2small")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--ell", type=int, default=24)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open("configs/e0.yaml"))
    seed = 9001
    C, H = 336, 24
    panel = load_panel("eld")
    val = compute_validity(panel, C, H)
    pools = make_pools(panel, cfg["pools"], cfg["seed"])
    rng = np.random.default_rng(seed)
    vic = rng.permutation(pools["victim"])
    n_t = int(round(0.3 * len(vic)))
    targets, established = np.sort(vic[:n_t]), np.sort(vic[n_t:])
    service = np.sort(np.concatenate([established, pools["aux"]]))
    ok = day_mask(val, ["2013-01-15", "2013-12-20"], cfg["datasets"]["eld"]["exclude_days"])
    days = np.where(ok)[0]
    ctx = lambda u, d: panel.values[u, val.origins[d] - C : val.origins[d]].astype(np.float32)  # noqa: E731
    hor = lambda u, d: panel.values[u, val.origins[d] : val.origins[d] + H].astype(np.float32)  # noqa: E731
    kinds = list(dp.REPRESENTATIONS)
    Ps = [4, 8]
    epss = [math.inf, 4.0, 1.0]
    # public centres from public users on 40 random 2013 days
    cdays = rng.choice(days, size=40, replace=False)
    centres = {}
    for kind in kinds:
        Rp = np.stack([dp.represent(ctx(u, d), kind) for d in cdays for u in pools["public"] if val.valid[u, d]])
        for P in Ps:
            centres[(kind, P)] = dp.fit_centres(Rp, P, seed)
    fc = Forecaster(cfg["models"][a.model])
    groups, meta = [], []
    rel_cache: dict = {}
    for i in range(a.n):
        d = int(rng.choice(days))
        tv = targets[val.valid[targets, d]]
        sv = service[val.valid[service, d]]
        u = int(rng.choice(tv))
        cot = rng.choice(sv, size=a.k - 1, replace=False)
        tgt = pad_left(ctx(u, d)[-a.ell :], C)
        groups.append(tgt[None])
        meta.append((i, "iso", None))
        groups.append(np.stack([tgt] + [ctx(int(c), d) for c in cot]))
        meta.append((i, f"xl{a.k}", None))
        for kind in kinds:
            R = np.stack([dp.represent(ctx(int(s), d), kind) for s in sv])
            Cn = dp.default_clip(kind, C)
            for P in Ps:
                for eps in epss:
                    key = (d, kind, P, eps)
                    if key not in rel_cache:
                        r = np.random.default_rng(stable_seed("pilot", str(d), kind, str(P), str(eps), base=seed))
                        rel_cache[key] = dp.release(R, centres[(kind, P)], eps, 1e-5, Cn, r)
                    rel = rel_cache[key]
                    groups.append(np.stack([tgt] + [dp.to_series(p, kind, C) for p in rel.prototypes]))
                    meta.append((i, f"d4_{kind}_P{P}_eps{eps}", rel.counts.min()))
        meta_y = hor(u, d)
        meta.append((i, "y", meta_y))
    t0 = time.time()
    model_groups = [g for g, m in zip(groups, [m for m in meta if m[1] != "y"])]
    preds = fc.predict_groups(model_groups)
    dur = time.time() - t0
    ys = {m[0]: m[2] for m in meta if m[1] == "y"}
    rows: dict[str, list[float]] = {}
    min_counts: dict[str, list[float]] = {}
    for p, m in zip(preds, [m for m in meta if m[1] != "y"]):
        rows.setdefault(m[1], []).append(wql_item(ys[m[0]], p[0]))
        if m[2] is not None:
            min_counts.setdefault(m[1], []).append(float(m[2]))
    iso = np.array(rows["iso"])
    out = {"model": a.model, "n": a.n, "k_xl": a.k, "ell": a.ell, "period": "ELD 2013 (training period)", "seconds": round(dur, 1),
           "n_service_median": int(np.median([val.valid[service, d].sum() for d in days])), "results": {}}
    for name, w in rows.items():
        w = np.array(w)
        ok_ = np.isfinite(w) & np.isfinite(iso)
        out["results"][name] = {"wql": float(w[ok_].mean()), "gain_vs_iso": float(1 - w[ok_].mean() / iso[ok_].mean()),
                                "min_cluster_size_median": float(np.median(min_counts[name])) if name in min_counts else None}
    xl = out["results"][f"xl{a.k}"]["gain_vs_iso"]
    for name, r in out["results"].items():
        r["retained_vs_xl"] = r["gain_vs_iso"] / xl if xl > 0 else None
    json.dump(out, open(a.out, "w"), indent=2)
    for name, r in sorted(out["results"].items(), key=lambda kv: -kv[1]["gain_vs_iso"]):
        print(f"{name:32} gain {r['gain_vs_iso']:+.4f} retained {r['retained_vs_xl'] if r['retained_vs_xl'] is None else round(r['retained_vs_xl'], 2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
