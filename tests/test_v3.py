"""Tests for the v0.3 confirmatory study (equivalence TOST, stronger sybil attacker, config/report wiring)."""

from __future__ import annotations

import numpy as np
import yaml

from crossleak.analysis import v3_report as V3
from crossleak.experiments import main as Mx
from crossleak.experiments import registry

CFG = yaml.safe_load(open("configs/v3.yaml"))
CFG["_sha256"] = "test"


def test_v3_driver_reuses_main_cells():
    assert registry.driver_name(CFG) == "v3"
    cells = registry.build_cells(CFG)
    # same cell objects as the main driver
    assert cells == Mx.build_cells(CFG)
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))
    exps = {c.exp for c in cells}
    assert {"util", "attack", "channel", "ref"} <= exps
    assert not any(c.exp == "cost" for c in cells)  # cost models empty in v3


def test_v3_shards_balanced_and_fit_overnight():
    cells = registry.build_cells(CFG)
    a = registry.assign_shards(cells, CFG, 5)
    load = [sum(c.cost(CFG) for c in cells if a[c.cell_id] == [s] and c.exp != "cost") for s in range(5)]
    assert max(load) / min(load) < 1.05
    assert sum(load) / 3600 < 30  # well under one overnight run on five machines


def test_v3_attack_has_larger_training_sets():
    cells = registry.build_cells(CFG)
    sizes = {int(k): v for k, v in CFG["attack"]["sizes"].items()}
    assert sizes[256][0] >= 1500 and sizes[64][0] >= 2000  # stronger than v0.2 (800 / 1000)
    ks = {c.p["k"] for c in cells if c.exp == "attack" and c.p["fam"] == "sybil"}
    assert {16, 64, 256} <= ks


def _fake_util_results(delta_gain: float, n=600, seed=0):
    """Two configs a,b whose true gain differs by delta_gain, plus iso, as one util cell result + arrays."""
    rng = np.random.default_rng(seed)
    iso = rng.uniform(0.08, 0.12, n).astype(float)
    base = 0.10
    wa = iso * (1 - base)
    wb = iso * (1 - (base - delta_gain))  # gain(a) - gain(b) ~ delta_gain
    users = rng.integers(0, 40, n)
    import tempfile, os, json

    d = tempfile.mkdtemp()
    np.savez(os.path.join(d, "arrays.npz"), wql_iso=iso, wql_d1_16=wa, wql_xl16=wb, user=users)
    res = {"cell_id": "util__eld__c2base__d1-16__d4-False__ell-24__part-0", "exp": "util", "dataset": "eld",
           "model": "c2base", "params": {"ell": 24, "part": 0, "n": n}, "status": "ok", "_dir": d,
           "summary": {}, "config_sha256": "test"}
    return {res["cell_id"]: res}


def test_equivalence_tost_detects_equivalent_and_not():
    cfg = {"equivalence": {"margin": 0.02, "model": "c2base",
                           "pairs": [{"dataset": "eld", "ell": 24, "a": "d1_16", "b": "xl16"}], "no_benefit": []}}
    close = V3._equivalence(_fake_util_results(0.0, seed=1), cfg, 500)
    assert close["pairs"][0]["verdict"] == "EQUIVALENT"
    far = V3._equivalence(_fake_util_results(0.06, seed=2), cfg, 500)
    assert far["pairs"][0]["verdict"] == "NOT_EQUIVALENT"


def test_report_verdicts_partial_and_mismatch():
    V = V3._verdicts(CFG, {"pairs": [], "no_benefit": []}, {}, {}, {"invalid_share": 0.0, "missing": 3, "config_mismatch": 0})
    assert V["overall"] == "PARTIAL"
    V = V3._verdicts(CFG, {"pairs": [], "no_benefit": []}, {}, {}, {"invalid_share": 0.0, "missing": 0, "config_mismatch": 1})
    assert V["overall"] == "CONFIG_MISMATCH"


def test_agcheck_script_importable_and_frames():
    # the agcheck script must parse and expose its feature layout without autogluon present
    import importlib.util
    import pathlib

    p = pathlib.Path("scripts/agcheck.py")
    spec = importlib.util.spec_from_file_location("agcheck", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.QLEVELS == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    assert m.H == 24 and m.C == 336
