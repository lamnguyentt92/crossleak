"""Verdict rules, report generation, artifact packaging and (optional) real-model conformance."""

from __future__ import annotations

import json
import os
import zipfile

import pytest

from crossleak.analysis import e0_report


def _write(root, cell_id, exp, ds, model, params, summary, status="ok"):
    d = root / cell_id
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"cell_id": cell_id, "exp": exp, "dataset": ds, "model": model, "params": params, "summary": summary, "status": status, "host": "t"}))
    (d / "DONE").write_text("{}")


def _gain(est, lo, hi):
    return {"n_targets": 10, "n_users": 5, "wql_iso_mean": 1, "wql_xl_mean": 1, "wql_full_mean": 1, "wql_snaive_mean": 1, "gain": {"est": est, "lo": lo, "hi": hi}, "share_items_improved": 0.5, "mase_iso_mean": 1, "mase_xl_mean": 1}


def _dec(dr_lo, dauc_lo):
    b = lambda lo: {"est": lo + 0.02, "lo": lo, "hi": lo + 0.04}  # noqa: E731
    return {"n_test": 10, "n_test_victims": 5, "r_median_ctrl": 0.1, "r_median_xl": 0.2, "r_median_perm": 0.0, "delta_r": b(dr_lo), "auc_ctrl": 0.5, "auc_xl": b(0.6), "auc_perm": 0.5, "delta_auc": b(dauc_lo)}


@pytest.mark.parametrize(
    "gain_lo,gain_est,dr_lo,dauc_lo,expected",
    [
        (0.01, 0.08, 0.10, -0.1, "GO"),
        (0.001, 0.03, 0.10, -0.1, "GO_REFRAME"),
        (0.01, 0.08, 0.01, -0.1, "WARN_EXTEND"),
        (0.01, 0.08, -0.02, -0.03, "STOP"),
    ],
)
def test_verdict_table(tmp_path, gain_lo, gain_est, dr_lo, dauc_lo, expected):
    _write(tmp_path, "g1", "gain", "eld", "c2base", {"ell": 24, "k": 4, "scen": "mixed"}, _gain(gain_est, gain_lo, 0.2))
    _write(tmp_path, "g2", "gain", "eld", "c2base", {"ell": 168, "k": 4, "scen": "mixed"}, _gain(0.5, 0.4, 0.6))  # ell>72 must be ignored
    _write(tmp_path, "d1", "decode", "eld", "c2small", {"k": 2, "probe": "short"}, _dec(dr_lo, dauc_lo))
    results, failed = e0_report.load_results([tmp_path])
    v = e0_report.verdicts(results, failed, n_planned=3)
    assert v["overall"] == expected
    assert e0_report.verdicts(results, failed, n_planned=10)["overall"] == f"PARTIAL_{expected}"


def test_invalid_fraction_rule(tmp_path):
    _write(tmp_path, "d1", "decode", "eld", "c2small", {"k": 2, "probe": "short"}, _dec(0.2, 0.2))
    for i in range(3):
        d = tmp_path / f"f{i}"
        d.mkdir()
        (d / "FAILED").write_text(json.dumps({"attempts": 2, "error": "boom"}))
    results, failed = e0_report.load_results([tmp_path])
    assert e0_report.verdicts(results, failed, n_planned=10)["overall"] == "NOT_INTERPRETABLE"


def test_report_and_artifact(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    root = runs / "e0"
    _write(root, "g1", "gain", "eld", "c2base", {"ell": 24, "k": 4, "scen": "mixed"}, _gain(0.08, 0.01, 0.2))
    _write(root, "d1", "decode", "eld", "c2small", {"k": 2, "probe": "short"}, _dec(0.1, 0.1))
    for var, sub in (("CROSSLEAK_RUNS", "runs"), ("CROSSLEAK_REPORTS", "reports"), ("CROSSLEAK_LOGS", "logs"), ("CROSSLEAK_ARTIFACTS", "art"), ("CROSSLEAK_DATA", "data")):
        monkeypatch.setenv(var, str(tmp_path / sub))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "x.log").write_text("hello")
    path = e0_report.build_report([root], "e0", n_planned=2)  # both planned cells present
    text = path.read_text()
    assert "Overall: GO" in text
    from crossleak.artifact import make_artifact

    z = make_artifact("unit")
    from crossleak.config import TAG

    assert z.name.startswith(f"crossleak_{TAG}_") and z.name.endswith("_unit.zip")
    names = zipfile.ZipFile(z).namelist()
    assert any(n.endswith("runs/e0/g1/result.json") for n in names)
    assert any(n.endswith("reports/e0/E0_REPORT.md") for n in names)
    assert any(n.endswith("env/pip_freeze.txt") for n in names)
    assert any(n.endswith("prereg/E0_PREREG.md") for n in names)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CROSSLEAK_SLOW") != "1", reason="set CROSSLEAK_SLOW=1 to download models")
def test_real_model_conformance():
    from crossleak.doctor import conformance

    res = conformance({"c2small": "autogluon/chronos-2-small"})
    assert res["c2small"]["pass"], res
