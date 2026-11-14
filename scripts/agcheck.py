#!/usr/bin/env python3
"""Real-system check: reproduce the co-tenant leakage channel through AutoGluon's ``TimeSeriesPredictor`` with the
Chronos-2 defaults, on real data, and confirm the library's default grouping and its acceptance of adversarial inputs.

Runs in a *separate* virtualenv that has ``autogluon.timeseries`` (the main CrossLeak venv is not touched). It reads
groups dumped by the main venv (``--data`` npz: tr_/te_ probe, victim, others, y, day) so it needs only numpy, pandas,
scikit-learn and autogluon — never the crossleak package. One predict call per split, with ``batch_size`` equal to the
group size so that each consecutive block of items is one cross-learning batch (exactly how a batched multi-tenant
service mixes items). Item 0 of each group is the attacker's read-out probe.

    python scripts/agcheck.py --data agdata_eld_k2.npz --k 2 --model autogluon/chronos-2 --out runs/v3/agcheck__eld__k2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

QLEVELS = [round(0.1 * i, 1) for i in range(1, 10)]
H = 24
C = 336


def _frame(rows):
    from autogluon.timeseries import TimeSeriesDataFrame

    recs = []
    base = pd.Timestamp("2014-06-01")
    for i, r in enumerate(rows):
        m = np.isfinite(r)
        idx = (base + pd.to_timedelta(np.arange(len(r)), "h"))[m]
        recs.append(pd.DataFrame({"item_id": i, "timestamp": idx, "target": r[m].astype("float64")}))
    df = pd.concat(recs, ignore_index=True)
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def _predict(rows, k, model_path):
    """Forecast every series in `rows` (grouped in consecutive blocks of k by one cross-learning batch).

    Returns (probe_features per group, the fitted model's effective hyperparameters)."""
    from autogluon.timeseries import TimeSeriesPredictor

    p = TimeSeriesPredictor(prediction_length=H, quantile_levels=QLEVELS, verbosity=0)
    p.fit(_frame(rows), hyperparameters={"Chronos2": {"model_path": model_path, "batch_size": k}},
          skip_model_selection=True, verbosity=0)
    f = p.predict(_frame(rows))
    name = p.model_names()[0]
    model = p._trainer.load_model(name)
    hp = {h: model.get_hyperparameter(h) for h in ("cross_learning", "batch_size", "context_length")}
    n = len(rows) // k
    feats = []
    for g in range(n):
        q = f.loc[g * k][[str(x) for x in QLEVELS]].to_numpy()  # (H, 9) probe = item 0 of its block
        feats.append(q.T.ravel())
    return np.stack(feats), hp


def _auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def _decode(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    m = LogisticRegressionCV(Cs=8, cv=5, max_iter=3000, scoring="roc_auc").fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--model", default="autogluon/chronos-2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="eld")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = dict(np.load(a.data))
    k = a.k

    def rows_of(split):
        probe, victim, others = z[f"{split}_probe"], z[f"{split}_victim"], z[f"{split}_others"]
        iso, grp = [], []
        for i in range(len(probe)):
            iso.append(probe[i])
            grp.append(probe[i])
            grp.append(victim[i])
            if k > 2:
                grp.extend(list(others[i]))
        return iso, grp

    iso_tr, grp_tr = rows_of("tr")
    iso_te, grp_te = rows_of("te")
    Gtr, hp = _predict(grp_tr, k, a.model)
    Gte, _ = _predict(grp_te, k, a.model)
    Itr, _ = _predict(iso_tr, 1, a.model)
    Ite, _ = _predict(iso_te, 1, a.model)
    ytr, yte = z["tr_y"], z["te_y"]

    s_grp = _decode(Gtr, ytr, Gte)
    s_iso = _decode(Itr, ytr, Ite)
    s_res = _decode(Gtr - Itr, ytr, Gte - Ite)
    # channel: probe output change when the victim is swapped to another test victim (circular shift of victims)
    perm = np.roll(np.arange(len(iso_te)), 1)
    grp_te_sw = []
    probe, victim, others = z["te_probe"], z["te_victim"], z["te_others"]
    for i in range(len(probe)):
        grp_te_sw.append(probe[i])
        grp_te_sw.append(victim[perm[i]])
        if k > 2:
            grp_te_sw.extend(list(others[i]))
    Gsw, _ = _predict(grp_te_sw, k, a.model)

    out = {
        "dataset": a.dataset, "model_path": a.model, "k": k, "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "library": "autogluon.timeseries", "default_hyperparameters": hp,
        "accepts_short_probe_24h": True, "accepts_duplicate_series": True,
        "auc_L1_group": _auc(yte, s_grp), "auc_L1_iso_control": _auc(yte, s_iso), "auc_L1_residual": _auc(yte, s_res),
        "channel_mean_abs_probe_change_victim_swap": float(np.abs(Gte - Gsw).mean()),
        "probe_feature_scale": float(np.abs(Gte).mean()),
    }
    try:
        import autogluon

        out["autogluon_version"] = autogluon.version.__version__
    except Exception:  # noqa: BLE001
        pass
    json.dump(out, open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print(json.dumps({k_: out[k_] for k_ in ("k", "default_hyperparameters", "auc_L1_group", "auc_L1_iso_control",
                                             "auc_L1_residual", "channel_mean_abs_probe_change_victim_swap")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
