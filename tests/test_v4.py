"""v0.4 causal-defence driver: runtime patching of the frozen main driver, past-only public windows, configuration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from crossleak.config import load_config
from crossleak.experiments import main as M
from crossleak.experiments import registry, v4

ROOT = Path(__file__).resolve().parents[1]


def test_install_and_uninstall_restore_the_main_driver():
    orig = (M.build_cells, M.Env, M.RUNNERS["util"])
    v4.install()
    try:
        assert M.build_cells is v4.build_cells and M.Env is v4.CausalEnv and M.RUNNERS["util"] is v4.run_util
        v4.install()  # idempotent
        assert issubclass(M.Env, orig[1])
    finally:
        v4.uninstall()
    assert (M.build_cells, M.Env, M.RUNNERS["util"]) == orig


def test_registry_installs_per_driver_and_v4_builds_only_utility_cells():
    cfg = load_config(str(ROOT / "configs" / "v4.yaml"))
    try:
        cells = registry.build_cells(cfg)
        assert cells and {c.exp for c in cells} == {"util"}
        assert M.Env is v4.CausalEnv
        registry.install_driver({"driver": "v3"})
        assert M.Env is v4._ORIGINAL["Env"]
    finally:
        v4.uninstall()


def test_past_only_windows_never_reach_the_origin_or_the_future():
    D, gap = 200, 56
    ds = SimpleNamespace(D=D, weekday=np.arange(D) % 7)
    pub = np.arange(3)
    valid = np.ones((3, D), dtype=bool)
    for d in (60, 120, 199):
        cand, days = v4._public_windows(ds, pub, valid, d, gap, past_only=True)
        assert len(days) and (days <= d - gap).all() and (ds.weekday[days] == ds.weekday[d]).all()
        _, both = v4._public_windows(ds, pub, valid, d, gap, past_only=False)
        assert set(days) <= set(both) and (np.abs(both - d) >= gap).all()
    cand, days = v4._public_windows(ds, pub, valid, 40, gap, past_only=True)  # too early: nothing in the past
    assert len(days) == 0 and cand == []


def test_v4_configuration_names_match_the_runner():
    cfg = load_config(str(ROOT / "configs" / "v4.yaml"))
    assert cfg["driver"] == "v4" and cfg["equivalence"]["margin"] == 0.02 and cfg["equivalence"]["model"] == "c2base"
    names = {f"xl{k}" for u in cfg["utility"] for k in u["xl_k"]}
    names |= {f"d1_{k}" for u in cfg["utility"] for k in u["d1_k"]} | {v4.d1p_name(k) for u in cfg["utility"] for k in u["d1_k"]}
    names |= {M.d4_name(P, e) for P in cfg["d4"]["P"] for e in cfg["d4"]["eps"]}
    names |= {v4.d4p_name(P, e) for P in cfg["d4"]["P"] for e in cfg["d4"]["eps"]}
    for spec in cfg["equivalence"]["pairs"] + cfg["effects"]:
        assert spec["a"] in names and spec["b"] in names, spec
    assert v4.d4p_name(8, 1) == "d4p_P8_eps1" and v4.past_centres_datasets(cfg) == ["lifesnaps"]


def test_frozen_main_driver_is_unchanged():
    line = next(l for l in (ROOT / "prereg" / "MAIN_FREEZE.sha256").read_text().splitlines()
                if l.endswith("src/crossleak/experiments/main.py"))
    digest = line.split()[0]
    assert hashlib.sha256((ROOT / "src/crossleak/experiments/main.py").read_bytes()).hexdigest() == digest


def test_watch_detects_the_running_experiment(tmp_path):
    """Regression (v0.4.0): the dashboard's auto-detection used `glob` without importing it."""
    import importlib.util
    import os
    import time

    spec = importlib.util.spec_from_file_location("watch", ROOT / "scripts" / "watch.py")
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "main.yaml").write_text("experiment: main\n")
    assert w.detect_config(str(tmp_path)) == "configs/main.yaml"
    for exp in ("v3", "v4"):
        d = tmp_path / "runs" / exp
        d.mkdir(parents=True)
        (d / "plan_main_shard0of5.json").write_text("{}")
        os.utime(d / "plan_main_shard0of5.json", (time.time() + (1 if exp == "v4" else 0),) * 2)
    assert w.detect_config(str(tmp_path)) == "configs/v4.yaml"
