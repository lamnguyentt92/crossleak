"""Report of the v0.4 causal-defence study (prereg/V4_PREREG.md, frozen by prereg/V4_FREEZE.sha256).

Reuses the frozen analysis helpers: ``main_report.analyse_utility`` (gains with cluster bootstrap) and
``v3_report._equivalence`` (TOST on paired gain differences). Verdicts EQC_* are the confirmatory rules of the plan;
the effect of causality (d1p - d1, d4p - d4) is an estimand.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from crossleak.analysis import main_report as M
from crossleak.analysis.v3_report import _equivalence
from crossleak.config import TAG

log = logging.getLogger("crossleak.v4report")


def _verdicts(cfg: dict, equiv: dict, effects: dict, coverage: dict) -> dict:
    V: dict = {}
    for row in equiv["pairs"]:
        V[f"EQC_{row['dataset']}_{row['a']}_vs_{row['b']}"] = row["verdict"]
    if coverage["invalid_share"] > 0.10:
        V["overall"] = "NOT_INTERPRETABLE"
    elif coverage["missing"] or coverage["failed"]:
        V["overall"] = "PARTIAL"
    else:
        V["overall"] = "COMPLETE"
    if coverage.get("config_mismatch"):
        V["overall"] = "CONFIG_MISMATCH"
    return V


def build_report(roots: list[Path], cfg: dict, decoders: str = "none", n_jobs: int = 1, out_dir: Path | None = None) -> Path:
    from crossleak.config import reports_dir
    from crossleak.experiments.registry import build_cells

    t0 = time.time()
    exp = cfg["experiment"]
    results, failed = M.load_results(roots)
    planned = {c.cell_id for c in build_cells(cfg)}
    ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
    infeasible = sorted(k for k, r in results.items() if r.get("status") == "infeasible")
    missing = sorted(planned - set(results))
    mismatch = sorted(k for k, r in results.items() if cfg.get("_sha256") and r.get("config_sha256") != cfg["_sha256"])
    coverage = {"planned": len(planned), "ok": len(ok), "infeasible": len(infeasible), "failed": len(failed),
                "missing": len(missing), "invalid_share": (len(failed) + len(infeasible)) / max(1, len(planned)),
                "config_mismatch": len(mismatch), "infeasible_cells": infeasible, "failed_cells": [f["cell_id"] for f in failed],
                "missing_cells": missing[:200]}
    n_boot = int(cfg["bootstrap"])
    util = M.analyse_utility(results, cfg, n_boot)
    equiv = _equivalence(results, cfg, n_boot)
    effects = _equivalence(results, {"equivalence": {**cfg["equivalence"], "pairs": cfg["effects"], "no_benefit": []}}, n_boot)
    V = _verdicts(cfg, equiv, effects, coverage)
    summary = {"version": TAG, "experiment": exp, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "freeze_v4": M.check_freeze("V4_FREEZE.sha256"), "coverage": coverage, "verdicts": V,
               "equivalence": equiv, "effects": effects, "utility": util, "runtime_min": round((time.time() - t0) / 60, 1)}
    out = out_dir or (reports_dir() / exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v4_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    (out / "V4_REPORT.md").write_text(_markdown(summary))
    log.info("v4 report written to %s (%.1f min)", out, (time.time() - t0) / 60)
    return out / "V4_REPORT.md"


def _pp(x) -> str:
    return "n/a" if x is None else f"{100 * x:+.2f}"


def _markdown(s: dict) -> str:
    L = [f"# CrossLeak {s['version']} — causal defences (v0.4)", "",
         f"Freeze: **{s['freeze_v4'].get('status')}**. Coverage: {s['coverage']['ok']}/{s['coverage']['planned']} cells ok, "
         f"{s['coverage']['failed']} failed, {s['coverage']['infeasible']} infeasible, {s['coverage']['missing']} missing. "
         f"Overall: **{s['verdicts']['overall']}**.", "",
         "## Confirmatory: equivalence with cross-learning (TOST, 90 % CI, margin ±2 pp)", "",
         "| Dataset | a | b | Δgain (pp) | 90 % CI | Verdict |", "|---|---|---|---|---|---|"]
    for r in s["equivalence"]["pairs"]:
        L.append(f"| {r['dataset']} | {r['a']} | {r['b']} | {_pp(r.get('delta'))} | [{_pp(r.get('lo90'))}, {_pp(r.get('hi90'))}] | "
                 f"{r['verdict']} |")
    L += ["", "## Estimand: effect of restricting public data to the past (90 % CI)", "",
          "| Dataset | causal | original | Δgain (pp) | 90 % CI |", "|---|---|---|---|---|"]
    for r in s["effects"]["pairs"]:
        L.append(f"| {r['dataset']} | {r['a']} | {r['b']} | {_pp(r.get('delta'))} | [{_pp(r.get('lo90'))}, {_pp(r.get('hi90'))}] |")
    L += ["", "## Gains over isolated forecasting (95 % CI)", ""]
    for ds, by_model in s["utility"].items():
        for model, by_ell in by_model.items():
            for ell, u in by_ell.items():
                L.append(f"- {ds} {model} ℓ={ell} ({u['n_targets']} targets, {u['n_users']} users): " + "; ".join(
                    f"{c} {100 * v['gain']['est']:+.1f} [{100 * v['gain']['lo']:+.1f}, {100 * v['gain']['hi']:+.1f}]"
                    for c, v in u["configs"].items()))
    L += ["", f"Report runtime {s['runtime_min']} min."]
    return "\n".join(L) + "\n"
