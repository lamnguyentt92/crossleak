#!/usr/bin/env python3
"""Re-analysis of the E0 pilot decode cells with the main study's labels and metrics (L1, R1, A1), run BEFORE the
main freeze to calibrate its thresholds. Pilot data support no claim.

    python scripts/pilot_recalibrate_e0.py --e0 <dir with unpacked E0 machine artifacts> --out docs/pilots/e0_recalibration.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import yaml

from crossleak import attack as A
from crossleak.decode import evening_score
from crossleak.experiments.main import Cell, Env, run_ref
from crossleak.metrics import auc, cluster_bootstrap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0", required=True)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="c2base,c2small")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["_sha256"] = "pilot"
    env = Env(cfg)
    out = {}
    refs = {}
    for ds in ("eld", "lifesnaps"):
        _, refs[ds] = run_ref(Cell("ref", ds, "none", ()), env, 0)
    for f in sorted(glob.glob(os.path.join(a.e0, "*", "runs", "e0", "decode__*", "arrays.npz"))):
        name = os.path.basename(os.path.dirname(f))
        _, ds, model, kk, probe = name.split("__")
        if model not in a.models.split(","):
            continue
        z = dict(np.load(f))
        ref = refs[ds]
        Xtr, Ctr = A.features(z, "_tr")
        Xte, Cte = A.features(z, "_te")
        thr = float(np.median(evening_score(z["Y_tr"])))
        Ltr = A.labels(z["Y_tr"], None, z["day_tr"], ref["ref_even_med"], thr)
        Lte = A.labels(z["Y_te"], None, z["day_te"], ref["ref_even_med"], thr)
        dx = A.linear_decoder(Xtr, z["Y_tr"], Ltr, Xte, 1)
        dc = A.linear_decoder(Ctr, z["Y_tr"], Ltr, Cte, 1)
        rm = ref["ref_mean"][z["day_te"]]
        r_x, r_c = A.within_day_r(dx["yhat"], z["Y_te"], rm), A.within_day_r(dc["yhat"], z["Y_te"], rm)
        vic = z["victim_te"]
        row = {"n_test": int(len(vic))}
        for lab in ("L1", "L2"):
            row[f"auc_{lab}_xl"] = auc(Lte[lab], dx["scores"][lab])
            row[f"auc_{lab}_ctrl"] = auc(Lte[lab], dc["scores"][lab])
        ok = np.isfinite(r_x) & np.isfinite(r_c)
        row["r1_median_xl"] = float(np.nanmedian(r_x))
        row["r1_median_ctrl"] = float(np.nanmedian(r_c))
        row["delta_auc_L1"] = cluster_bootstrap(lambda i: auc(Lte["L1"][i], dx["scores"]["L1"][i]) - auc(Lte["L1"][i], dc["scores"]["L1"][i]), vic, 300, 2)
        row["delta_r1"] = cluster_bootstrap(lambda i: float(np.mean(r_x[ok][i] - r_c[ok][i])), vic[ok], 300, 3)
        victim_pool = ref["pools_victim"]
        for N in (10, 50):
            lk = A.linkage(dx["yhat"], vic, z["day_te"], ref["cand"].astype(np.float32), victim_pool, N, 5)
            lc = A.linkage(dc["yhat"], vic, z["day_te"], ref["cand"].astype(np.float32), victim_pool, N, 5)
            row[f"a1_top1_N{N}_xl"] = float(np.nanmean(lk["top1"])) if np.isfinite(lk["top1"]).any() else None
            row[f"a1_top1_N{N}_ctrl"] = float(np.nanmean(lc["top1"])) if np.isfinite(lc["top1"]).any() else None
            row[f"a1_n_N{N}"] = int(np.isfinite(lk["top1"]).sum())
        out[name] = row
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, dict)},
              "dAUC_L1", round(row["delta_auc_L1"]["est"], 3), [round(row["delta_auc_L1"]["lo"], 3), round(row["delta_auc_L1"]["hi"], 3)], flush=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
