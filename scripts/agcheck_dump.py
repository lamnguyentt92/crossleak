#!/usr/bin/env python3
"""Dump real co-tenant groups for the AutoGluon real-system check (run in the MAIN venv; see docs/AGCHECK.md).

Writes an npz with tr_/te_ arrays (probe, victim, others, y, day, vid) that scripts/agcheck.py (run in the AutoGluon
venv) reads. Uses the frozen pools, validity and within-day label exactly as the attack cells do.

    python scripts/agcheck_dump.py --dataset eld --k 2 --n-train 250 --n-test 150 --out agdata_eld_k2.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import yaml

from crossleak.decode import zprofile
from crossleak.experiments.main import Env, make_probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3.yaml")
    ap.add_argument("--dataset", default="eld")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=250)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--probe", default="noise")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["_sha256"] = "agdump"
    env = Env(cfg)
    ds = env.ds(a.dataset)
    R = env.refs(a.dataset)
    rng = np.random.default_rng(a.seed)
    k = a.k

    def build(split, nvic):
        pool_v = "aux" if split == "train" else "victim"
        days = ds.feasible(split, {"aux": 1, pool_v: 1, "all": k + 2})
        probe_ctx, vic_ctx, oth, y, dd, vid = [], [], [], [], [], []
        tries = 0
        while len(y) < nvic and tries < 80 * nvic:
            tries += 1
            d = int(rng.choice(days))
            av = ds.users("aux", d)
            if len(av) < 1:
                continue
            at = int(rng.choice(av))
            vc = ds.users(pool_v, d)
            vc = vc[vc != at]
            if len(vc) < 1:
                continue
            v = int(rng.choice(vc))
            others = ds.users("all", d)
            others = others[(others != at) & (others != v)]
            if len(others) < k - 2:
                continue
            ot = rng.choice(others, size=k - 2, replace=False) if k > 2 else np.zeros(0, int)
            z = zprofile(ds.ctx(v, d)[-24:])
            if not np.isfinite(z).all():
                continue
            probe = make_probe(a.probe, ds.ctx(at, d), rng, ds.C, 24)
            probe_ctx.append(probe)
            vic_ctx.append(ds.ctx(v, d))
            oth.append(np.stack([ds.ctx(int(o), d) for o in ot]) if k > 2 else np.zeros((0, ds.C), np.float32))
            med = R["ref_even_med"][d] if split == "test" else R["ref_even_med_att"][d]
            y.append(int(z[18:24].mean() > med))
            dd.append(d)
            vid.append(v)
        return dict(probe=np.stack(probe_ctx), victim=np.stack(vic_ctx), others=np.stack(oth),
                    y=np.array(y), day=np.array(dd), vid=np.array(vid))

    tr, te = build("train", a.n_train), build("test", a.n_test)
    np.savez(a.out, **{f"tr_{x}": tr[x] for x in tr}, **{f"te_{x}": te[x] for x in te})
    print(f"{a.out}: k={k} train={len(tr['y'])} test={len(te['y'])} test_label_rate={te['y'].mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
