"""v0.5 driver: runtime patching, cell list, ablation windows, placement emulation, baselines and report rules."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from crossleak.analysis import v5_report as R
from crossleak.config import load_config
from crossleak.experiments import main as M
from crossleak.experiments import registry, v4, v5

ROOT = Path(__file__).resolve().parents[1]


def test_install_and_uninstall_restore_the_main_driver():
    orig = (M.build_cells, M.RUNNERS["util"], M.RUNNERS["cost"], M._short, M.Env)
    v5.install()
    try:
        assert M.build_cells is v5.build_cells and M.RUNNERS["util"] is v5.run_util and M.RUNNERS["cost"] is v5.run_serve
        assert M.RUNNERS["placement"] is v5.run_placement and M._short is v5._short
        v5.install()  # idempotent
    finally:
        v5.uninstall()
    assert (M.build_cells, M.RUNNERS["util"], M.RUNNERS["cost"], M._short, M.Env) == orig
    assert "placement" not in M.RUNNERS


def test_registry_switches_between_v4_and_v5():
    cfg = load_config(str(ROOT / "configs" / "v5.yaml"))
    try:
        registry.install_driver(cfg)
        assert M.build_cells is v5.build_cells and M.Env is v4._ORIGINAL["Env"]
        registry.install_driver({"driver": "v4"})
        assert M.Env is v4.CausalEnv and M.RUNNERS["cost"] is v5._ORIGINAL["cost"]
        registry.install_driver({"driver": "main"})
        assert M.build_cells is v5._ORIGINAL["build_cells"] and M.Env is v4._ORIGINAL["Env"]
    finally:
        v5.uninstall()
        v4.uninstall()


def test_cells_of_the_frozen_configuration():
    cfg = load_config(str(ROOT / "configs" / "v5.yaml"))
    try:
        cells = registry.build_cells(cfg)
        exps = {c.exp for c in cells}
        assert exps == {"util", "attack", "ref", "placement", "cost"}
        assert cfg["placement"]["policies"] == ["ordered", "informed", "informed_pad", "hashed"]
        att = [c for c in cells if c.exp == "attack"]
        # five realisations at k = 256 and k = 32, one half-sybil configuration; train and test sizes as v0.3 S1
        keys = {(c.p["fam"], c.p["k"], c.p["rep"]) for c in att}
        assert keys == {("sybil", 256, r) for r in range(1, 6)} | {("sybil", 32, r) for r in range(1, 6)} | {("half", 256, 1)}
        n = {}
        for c in att:
            n[(c.p["fam"], c.p["k"], c.p["rep"], c.p["role"])] = n.get((c.p["fam"], c.p["k"], c.p["rep"], c.p["role"]), 0) + c.p["n"]
        assert n[("sybil", 256, 3, "train")] == 1500 and n[("sybil", 256, 3, "test")] == 500
        assert n[("sybil", 32, 1, "train")] == 2000 and n[("half", 256, 1, "test")] == 500
        # realisations differ only by rep, so every realisation has its own seeds
        a1 = [c for c in att if c.p["rep"] == 1 and c.p["k"] == 256 and c.p["fam"] == "sybil"][0]
        a2 = [c for c in att if c.p["rep"] == 2 and c.p["k"] == 256 and c.p["fam"] == "sybil" and c.p["part"] == a1.p["part"]
              and c.p["role"] == a1.p["role"]][0]
        assert a1.sample_key != a2.sample_key and a1.family_key != a2.family_key
        util = [c for c in cells if c.exp == "util"]
        assert {c.dataset for c in util} == {"eld", "lifesnaps"} and all(c.p["arms"] == ("d1p", "d1x", "d1r", "d1o") for c in util)
        assert sum(c.p["n"] for c in util if c.dataset == "eld") == 600
        assert all(c.cost(cfg) <= cfg["part_budget_sec"] * 1.05 + 30 for c in cells if c.exp in ("util", "attack"))
        assert [c.exp for c in cells if M.phase_of(c) == "cost"] == ["cost"]
    finally:
        v5.uninstall()


def test_ablation_windows_respect_weekday_and_the_past():
    D, gap = 200, 56
    ds = SimpleNamespace(D=D, weekday=np.arange(D) % 7)
    pub = np.arange(4)
    valid = np.ones((4, D), dtype=bool)
    for d in (70, 150, 199):
        _, same = v5._windows(ds, pub, valid, d, gap, "same")
        _, other = v5._windows(ds, pub, valid, d, gap, "other")
        _, anyw = v5._windows(ds, pub, valid, d, gap, "any")
        assert (same <= d - gap).all() and (other <= d - gap).all() and (anyw <= d - gap).all()
        assert (ds.weekday[same] == ds.weekday[d]).all() and (ds.weekday[other] != ds.weekday[d]).all()
        assert set(same) | set(other) == set(anyw) and not set(same) & set(other)


def test_placement_policies():
    rng = np.random.default_rng(0)
    # hashed order: the attacker's share of a block is about m / (N + m)
    r = v5.simulate_policy("hashed", 10000, 10, 1000, 256, 30, rng)
    share = np.mean(r["all"]["share_if_co"])
    assert 0.06 < share < 0.13 and r["all"]["isolated_255"] == 0 and r["all"]["n"] == 10000 + r["n_new"]
    # registration order with a random burst: the burst sits in its own blocks; only clients next to it are reached
    r = v5.simulate_policy("ordered", 10000, 10, 255, 256, 30, rng)
    assert r["all"]["co_located"] <= 2 * 256 and r["new"]["n"] == r["n_new"]
    # informed attacker with enough meters: each 255-meter burst isolates exactly one new client
    r = v5.simulate_policy("informed", 1000 * 256, 10, 255 * 3, 256, 30, rng)
    assert r["new"]["isolated_255"] == 3 and r["attacker_meters"] == 255 * 3
    # informed attacker with fewer meters than a block: it still joins one block-opening new client, at share m / 256
    r = v5.simulate_policy("informed", 1000 * 256, 10, 64, 256, 30, rng)
    assert r["attacker_meters"] == 64 and r["new"]["co_located"] >= 1 and r["new"]["isolated_255"] == 0
    assert np.isclose(np.max(r["new"]["share_if_co"]), 64 / 256)
    # leftover budget after full blocks is used on the next block
    r = v5.simulate_policy("informed", 1000 * 256, 10, 300, 256, 30, rng)
    assert r["attacker_meters"] == 300
    s = v5.simulate_service_policy(1000, 16, 64, rng, 20000)
    assert abs(s["p_reach_given_victim"] - s["p_exact"]) < 0.02 and s["occupancy"] == 1 / 16


def test_informed_pad_pads_to_a_block_boundary_once():
    rng = np.random.default_rng(3)
    r = v5.simulate_policy("informed_pad", 10000, 1, 1000, 256, 30, rng)  # 10000 % 256 = 16: 240 padding meters first
    assert r["attacker_meters"] >= 240 and r["new"]["co_located"] >= 1
    r = v5.simulate_policy("informed_pad", 10000, 1, 100, 256, 30, rng)   # budget below the padding: no padding
    assert r["attacker_meters"] <= 100
    r = v5.simulate_policy("informed", 1000 * 256, 10, 255, 256, 30, rng)  # informed never pads
    assert r["new"]["isolated_255"] == 1


def test_half_sybil_group_has_256_members():
    k = 256
    m = M.n_attacker("half", k)
    assert m == 128 and 1 + 1 + (m - 1) + (k - 1 - m) == k  # probe + victim + copies + honest others


def test_last_partial_block_isolation_is_counted():
    a, size = v5._block_counts(np.arange(257, 300), np.array([256]), 300, 256)  # victim + 43 attackers in a 44-item block
    st = v5._victim_stats(a, size, 256)
    assert st["isolated_any"] == 1 and st["isolated_255"] == 0 and np.isclose(st["share_if_co"][0], 43 / 44)


def test_ridge_baseline_learns_a_daily_pattern():
    from crossleak import baselines as B

    rng = np.random.default_rng(1)
    t = np.arange(24)
    win = []
    for _ in range(800):
        lvl = rng.uniform(1, 10)
        day = lvl * (1.5 + np.sin(2 * np.pi * t / 24)) + 0.05 * rng.standard_normal(24)
        nxt = lvl * (1.5 + np.sin(2 * np.pi * t / 24)) + 0.05 * rng.standard_normal(24)
        win.append((day, nxt, int(rng.integers(0, 7)), int(rng.integers(0, 40))))
    fit = B._fit_one(win, "meanabs", 3)
    q = B.predict_ridge(fit, win[0][0], win[0][2])
    assert q.shape == (9, 24) and np.all(np.diff(q, axis=0) >= -1e-9)
    assert np.abs(q[4] - win[0][1]).mean() < 0.3 * np.abs(win[0][1]).mean()
    assert fit["cal_wql"] < 0.1


def test_report_rules():
    assert R.ablation_verdict(0.012, 0.03, 0.01) == "A_BETTER"
    assert R.ablation_verdict(0.002, 0.008, 0.01) == "A_BETTER_SMALL"
    assert R.ablation_verdict(-0.02, -0.011, 0.01) == "B_BETTER"
    assert R.ablation_verdict(-0.005, 0.009, 0.01) == "NO_DIFFERENCE"
    assert R.ablation_verdict(-0.02, 0.009, 0.01) == "INCONCLUSIVE"
    assert R.realisation_verdict(5, 5) == "STABLE_LEAK" and R.realisation_verdict(4, 5) == "STABLE_LEAK"
    assert R.realisation_verdict(3, 5) == "UNSTABLE" and R.realisation_verdict(0, 5) == "NOT_REPLICATED"
    assert R._rep_of("attack__eld__c2small__fam-sybil__filler-copy__k-256__probe-short__rep-4") == 4


def _resid(rep, lo, fam="sybil", k=256):
    b = lambda e, l_: {"est": e, "lo": l_, "hi": e + 0.05}  # noqa: E731
    return (f"attack__eld__c2small__fam-{fam}__filler-copy__k-{k}__probe-short__rep-{rep}",
            {"dataset": "eld", "model": "c2small", "fam": fam, "k": k, "auc_L1": b(0.6, 0.55), "auc_L1_ctrl": 0.5,
             "delta_auc_L1": b(lo + 0.05, lo), "delta_r1": b(0.1, 0.05), "n_train": 10, "n_test": 5})


def test_missing_realisations_count_as_not_positive_and_no_data_is_reported():
    planned = {("eld", "c2small", "sybil", 256): [1, 2, 3, 4, 5], ("eld", "c2small", "half", 256): [1]}
    resid = dict([_resid(1, 0.01), _resid(2, 0.02), _resid(3, 0.03), _resid(4, -0.01)])  # rep 5 missing
    out = R.analyse_realisations(resid, planned)
    syb = out["eld_c2small_sybil_k256"]
    assert syb["missing"] == [5] and syb["n_dauc_lo_pos"] == 3 and syb["verdict"] == "UNSTABLE"
    assert out["eld_c2small_half_k256"]["verdict"] == "NO_DATA"
    cov = {"config_mismatch": 0, "invalid_share": 0.0, "missing": 0, "failed": 0}
    V = R._verdicts({"contrasts": [], "gains": []}, out, [], cov, {"serve": {"models": ["c2base"]}})
    assert V["REP_eld_c2small_sybil_k256"] == "UNSTABLE_3of5" and V["HALF256"] == "NO_DATA" and V["SERVE_D4"] == "NO_DATA"
    assert V["overall"] == "PARTIAL"


def test_family_without_any_result_is_no_data_and_freeze_line():
    out = R.analyse_realisations({}, {("eld", "c2small", "sybil", 256): [1, 2, 3, 4, 5]})
    assert out["eld_c2small_sybil_k256"]["verdict"] == "NO_DATA"
    V = R._verdicts({"contrasts": [], "gains": []}, out, [], {"config_mismatch": 0, "invalid_share": 0.0, "missing": 0, "failed": 0},
                    {"serve": {"models": []}})
    assert V["REP_eld_c2small_sybil_k256"] == "NO_DATA" and V["overall"] == "PARTIAL"
    fz = {"status": "match", "files": {"a": {"expected": "x", "actual": "x", "match": True}, "b": {"expected": "x", "actual": "y", "match": False}}}
    assert R._freeze(fz) == "match (not matching: b)"


def test_frozen_main_driver_is_unchanged():
    want = {}
    for line in (ROOT / "prereg" / "V4_FREEZE.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            h, p = line.split()
            want[p] = h
    for p in ("src/crossleak/experiments/main.py", "src/crossleak/analysis/main_report.py", "src/crossleak/analysis/v3_report.py",
              "src/crossleak/defenses/dp_prototype.py", "src/crossleak/metrics.py", "src/crossleak/model.py"):
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == want[p], p
