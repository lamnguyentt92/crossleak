"""Post hoc (labelled): share of D4 prototypes that fall back to the public centre (noisy count < 1) on the days used
by the v0.3 utility cells. The releases are deterministic per day (seeded), so they are recomputed exactly.

    python docs/reanalysis/d4_fallback_share.py --runs 'unpacked/*/*/runs/v3' --config configs/v3.yaml
"""

import argparse
import glob

import numpy as np
import yaml

from crossleak.experiments.main import Env

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--config", default="configs/v3.yaml")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
cfg["_sha256"] = "posthoc"
env = Env(cfg)
for ds in ("eld", "lifesnaps"):
    days = set()
    for f in glob.glob(f"{a.runs}/util__{ds}__c2base__*ell-24*/arrays.npz"):
        days |= set(np.load(f)["day"].tolist())
    days = sorted(days)
    for eps in cfg["d4"]["eps"]:
        fb, cnt = [], []
        for d in days:
            rel = env.release(ds, int(d), int(cfg["d4"]["P_primary"]), float(eps))
            fb.append(rel.fallback.mean())
            cnt.append(rel.counts.sum())
        print(f"{ds:9} eps={eps}: days {len(days)}, service users per release (median) {np.median(cnt):.0f}, "
              f"prototypes on the public-centre fallback {np.mean(fb) * 100:.1f}%")
