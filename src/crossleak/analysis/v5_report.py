"""Report of the v0.5 study (prereg/V5_PREREG.md, frozen by prereg/V5_FREEZE.sha256).

Reuses the frozen analysis helpers: ``main_report.analyse_utility`` (gains, 95 % CI), ``v3_report._equivalence``
(paired gain differences, 90 % CI) and ``v3_report._residual_attacker`` (the confirmatory study's stronger attacker).
Verdicts:
  ABL_<id>   E5a contrasts: A_BETTER / B_BETTER (90 % CI above / below 0; *_SMALL if it also lies inside +/- margin),
             NO_DIFFERENCE (inside +/- margin, not significant), INCONCLUSIVE; EQ2_A5*: the v0.4 +/- 2-point rule;
             ABL_A4*: CONTEXT_HELPS / NO_GAIN_SHOWN for the other-domain gain over isolated forecasting.
  REP_<fam>_k<k>   E5c: STABLE_LEAK (at least 4 of the 5 *planned* realisations with a dAUC lower bound > 0; a missing
             realisation counts as not positive), UNSTABLE (1-3), NOT_REPLICATED (0).
  HALF256    E5d: LEAKS / NOT_DETECTED.
  SERVE_D4   E5b: MEETS_15MIN / EXCEEDS_15MIN (measured end-to-end time of D4 for 1,000 new clients).
  Every planned verdict without a result is NO_DATA, and the study is then PARTIAL.
Writes V5_REPORT.md, v5_summary.json, ablation.csv, realisations.csv, serve.csv, placement.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from crossleak.analysis import main_report as M
from crossleak.analysis.v3_report import _equivalence, _residual_attacker
from crossleak.config import TAG

log = logging.getLogger("crossleak.v5report")


def ablation_verdict(lo: float, hi: float, margin: float) -> str:
    """Plan section 3: superiority first; a significant difference whose CI lies inside +/- margin is *_SMALL."""
    inside = lo > -margin and hi < margin
    if lo > 0:
        return "A_BETTER_SMALL" if inside else "A_BETTER"
    if hi < 0:
        return "B_BETTER_SMALL" if inside else "B_BETTER"
    return "NO_DIFFERENCE" if inside else "INCONCLUSIVE"


def realisation_verdict(n_pos: int, n_planned: int) -> str:
    """Plan section 3: STABLE_LEAK if at least n_planned - 1 (4 of 5) planned realisations have a dAUC lower bound
    above 0; a realisation without a result counts as not positive."""
    if n_planned == 0:
        return "NO_DATA"
    if n_pos >= max(1, n_planned - 1):
        return "STABLE_LEAK"
    return "UNSTABLE" if n_pos > 0 else "NOT_REPLICATED"


def planned_families(cfg: dict) -> dict:
    """(dataset, model, fam, k) -> planned realisations, from the configuration."""
    out = {}
    for f in cfg["attack"]["families"]:
        for ds in f["datasets"]:
            for m in f["models"]:
                for k in f["k"]:
                    out[(ds, m, f["fam"], int(k))] = sorted(int(r) for r in f["reps"])
    return out


def _rep_of(key: str) -> int | None:
    m = re.search(r"__rep-(\d+)", key)
    return int(m.group(1)) if m else None


def analyse_ablation(results: dict, cfg: dict, n_boot: int) -> dict:
    ab = cfg["ablation"]
    eq = _equivalence(results, {"equivalence": {"margin": ab["margin"], "model": ab["model"], "pairs": ab["contrasts"],
                                                "no_benefit": ab.get("gains", [])}}, n_boot)
    margin = float(ab["margin"])
    rows = []
    for spec, row in zip(ab["contrasts"], eq["pairs"]):
        row = dict(row, id=spec["id"])
        row["tost_1pt"] = row.pop("verdict")
        row["rule"] = ablation_verdict(row["lo90"], row["hi90"], margin) if "lo90" in row else "NO_DATA"
        if spec["id"].startswith("A5"):  # the v0.4 E1c rule at +/- 2 points, reported next to A5
            row["eq_2pt"] = ("NO_DATA" if "lo90" not in row else
                             "EQUIVALENT" if (row["lo90"] > -0.02 and row["hi90"] < 0.02) else "NOT_EQUIVALENT")
        rows.append(row)
    gains = []
    for spec, row in zip(ab.get("gains", []), eq["no_benefit"]):
        row = dict(row, id=spec["id"])
        row.pop("verdict", None)
        row["rule"] = ("CONTEXT_HELPS" if row["lo90"] > 0 else "NO_GAIN_SHOWN") if "lo90" in row else "NO_DATA"
        gains.append(row)
    return {"margin": margin, "contrasts": rows, "gains": gains}


def analyse_realisations(resid: dict, planned: dict) -> dict:
    by: dict = defaultdict(list)
    for key, r in resid.items():
        if "auc_L1" not in r:
            continue
        by[(r["dataset"], r["model"], r["fam"], int(r["k"]))].append({
            "rep": _rep_of(key), "auc": r["auc_L1"]["est"], "auc_lo": r["auc_L1"]["lo"], "auc_hi": r["auc_L1"]["hi"],
            "ctrl": r.get("auc_L1_ctrl"), "dauc": r["delta_auc_L1"]["est"], "dauc_lo": r["delta_auc_L1"]["lo"],
            "dauc_hi": r["delta_auc_L1"]["hi"], "dr1": r["delta_r1"]["est"], "dr1_lo": r["delta_r1"]["lo"],
            "dr1_hi": r["delta_r1"]["hi"], "n_train": r["n_train"], "n_test": r["n_test"]})
    out = {}
    for (ds, model, fam, k), plan in sorted(planned.items()):
        reps = sorted(by.get((ds, model, fam, k), []), key=lambda x: (x["rep"] is None, x["rep"]))
        got = {x["rep"] for x in reps}
        missing = [r for r in plan if r not in got]
        a = np.array([x["auc"] for x in reps])
        d = np.array([x["dauc"] for x in reps])
        n_pos = int(sum(x["dauc_lo"] > 0 for x in reps))
        if not reps:
            verdict = "NO_DATA"
        elif fam == "sybil":
            verdict = realisation_verdict(n_pos, len(plan))
        else:
            verdict = "NO_DATA" if not reps else ("LEAKS" if n_pos > 0 else "NOT_DETECTED")
        st = lambda v, f: (float(f(v)) if len(v) else None)  # noqa: E731
        out[f"{ds}_{model}_{fam}_k{k}"] = {
            "dataset": ds, "model": model, "fam": fam, "k": int(k), "realisations": reps, "planned": plan, "missing": missing,
            "n": len(reps), "n_planned": len(plan), "n_dauc_lo_pos": n_pos,
            "auc_mean": st(a, np.mean), "auc_sd": float(a.std(ddof=1)) if len(a) > 1 else None, "auc_min": st(a, np.min),
            "auc_max": st(a, np.max), "dauc_mean": st(d, np.mean), "dauc_sd": float(d.std(ddof=1)) if len(d) > 1 else None,
            "dauc_min": st(d, np.min), "dauc_max": st(d, np.max), "verdict": verdict}
    return out


def _cell_summary(results: dict, exp: str) -> list[dict]:
    return [r for r in results.values() if r["exp"] == exp and r.get("status") == "ok"]


def _verdicts(abl: dict, reps: dict, serve: list[dict], coverage: dict, cfg: dict) -> dict:
    V: dict = {}
    for row in abl["contrasts"]:
        V[f"ABL_{row['id']}"] = row["rule"]
        if "eq_2pt" in row:
            V[f"EQ2_{row['id']}"] = row["eq_2pt"]
    for row in abl["gains"]:
        V[f"ABL_{row['id']}"] = row["rule"]
    for key, r in reps.items():
        if r["fam"] == "half" and r["k"] == 256:
            V["HALF256"] = r["verdict"]
        else:
            V[f"REP_{r['dataset']}_{r['model']}_{r['fam']}_k{r['k']}"] = (
                "NO_DATA" if r["verdict"] == "NO_DATA" else f"{r['verdict']}_{r['n_dauc_lo_pos']}of{r['n_planned']}")
    if cfg["serve"].get("models"):
        V["SERVE_D4"] = "NO_DATA"
    for s in serve:
        for row in s["summary"]["rows"]:
            if row["mode"] == "d4":
                V["SERVE_D4"] = "MEETS_15MIN" if row["total_s"] * 1000 / row["n_clients"] <= 900 else "EXCEEDS_15MIN"
    incomplete = any(r["missing"] for r in reps.values()) or any(v == "NO_DATA" for v in V.values())
    if coverage.get("config_mismatch"):
        V["overall"] = "CONFIG_MISMATCH"
    elif coverage["invalid_share"] > M.INVALID_MAX:
        V["overall"] = "NOT_INTERPRETABLE"
    else:
        V["overall"] = "COMPLETE" if (coverage["missing"] == 0 and coverage["failed"] == 0 and not incomplete) else "PARTIAL"
    return V


def build_report(roots: list[Path], cfg: dict, decoders: str = "linear", n_jobs: int = 1, out_dir: Path | None = None) -> Path:
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
    abl = analyse_ablation(results, cfg, n_boot)
    resid = _residual_attacker(results, n_boot, n_jobs) if decoders != "none" else {}
    reps = analyse_realisations(resid, planned_families(cfg))
    serve = _cell_summary(results, "cost")
    placement = _cell_summary(results, "placement")
    V = _verdicts(abl, reps, serve, coverage, cfg)
    summary = {"version": TAG, "experiment": exp, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "freeze_v5": M.check_freeze("V5_FREEZE.sha256"), "coverage": coverage, "verdicts": V, "utility": util,
               "ablation": abl, "attacks_residual": resid, "realisations": reps,
               "serve": [s["summary"] for s in serve], "placement": [p["summary"] for p in placement],
               "runtime_min": round((time.time() - t0) / 60, 1)}
    out = out_dir or (reports_dir() / exp)
    out.mkdir(parents=True, exist_ok=True)
    (out / "v5_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)))
    _csvs(out, summary)
    (out / "V5_REPORT.md").write_text(_markdown(summary, cfg))
    log.info("v5 report written to %s (%.1f min)", out, (time.time() - t0) / 60)
    return out / "V5_REPORT.md"


def _csvs(out: Path, s: dict) -> None:
    def write(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})

    write("ablation.csv", s["ablation"]["contrasts"] + s["ablation"]["gains"])
    write("realisations.csv", [dict(x, config=key) for key, r in s["realisations"].items() for x in r["realisations"]])
    write("serve.csv", [row for sv in s["serve"] for row in sv["rows"]])
    write("placement.csv", [row for p in s["placement"] for row in p["rows"]])
    write("service_policy.csv", [row for p in s["placement"] for row in p["service_policy"]])


def _freeze(fz: dict) -> str:
    bad = [p for p, v in (fz.get("files") or {}).items() if not (v.get("match") if isinstance(v, dict) else v in ("match", True))]
    return f"{fz.get('status')}" + (f" (not matching: {', '.join(bad)})" if bad else "")


def _n(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _f(x, s=100.0, nd=2):
    return "n/a" if x is None else f"{s * x:+.{nd}f}"


def _markdown(s: dict, cfg: dict) -> str:
    L = [f"# CrossLeak v0.5 report ({s['experiment']})", "",
         f"Version {s['version']}, created {s['created']}. Plan: prereg/V5_PREREG.md. Freeze check: {_freeze(s['freeze_v5'])}.", "",
         "## Verdicts", ""]
    L += [f"- {k}: **{v}**" for k, v in s["verdicts"].items()]
    c = s["coverage"]
    L += ["", f"Coverage: {c['ok']} of {c['planned']} cells ok, {c['failed']} failed, {c['infeasible']} infeasible, "
               f"{c['missing']} missing.", "", "## E5a public-context ablation (gain differences, points, 90 % CI)", "",
          "| ID | Dataset | a | b | a - b | 90 % CI | Rule | +/-2 rule (A5) |", "|---|---|---|---|---|---|---|---|"]
    for r in s["ablation"]["contrasts"]:
        L.append(f"| {r['id']} | {r['dataset']} | {r['a']} | {r['b']} | {_f(r.get('delta'))} | "
                 f"[{_f(r.get('lo90'))}, {_f(r.get('hi90'))}] | {r['rule']} | {r.get('eq_2pt', '')} |")
    for r in s["ablation"]["gains"]:
        L.append(f"| {r['id']} | {r['dataset']} | {r['config']} | iso | {_f(r.get('gain'))} | "
                 f"[{_f(r.get('lo90'))}, {_f(r.get('hi90'))}] | {r['rule']} | |")
    L += ["", "## E5c / E5d attacks over realisations (residual attacker, label L1)", "",
          "| Configuration | rep | AUC | 95 % CI | control | dAUC | 95 % CI | dR1 |", "|---|---|---|---|---|---|---|---|"]
    for key, r in s["realisations"].items():
        for x in r["realisations"]:
            L.append(f"| {key} | {x['rep']} | {_n(x['auc'])} | [{_n(x['auc_lo'])}, {_n(x['auc_hi'])}] | {_n(x['ctrl'])} | "
                     f"{_n(x['dauc'], sign=True)} | [{_n(x['dauc_lo'], sign=True)}, {_n(x['dauc_hi'], sign=True)}] | {_n(x['dr1'], sign=True)} |")
        L.append(f"| {key} summary | {r['n']} of {r['n_planned']} | mean {_n(r['auc_mean'])} | range {_n(r['auc_min'])}-{_n(r['auc_max'])} | "
                 f"missing {r['missing'] or 'none'} | mean {_n(r['dauc_mean'], sign=True)} | {r['n_dauc_lo_pos']} of {r['n_planned']} above 0 | "
                 f"**{r['verdict']}** |")
    L += ["", "## E5b end-to-end serving run (one machine)", "",
          "| Mode | clients | series/client | total s | prep s | call p50 s | call p95 s | clients/s | peak RSS MB | RSS increase MB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sv in s["serve"]:
        for r in sv["rows"]:
            L.append(f"| {r['mode']} | {r['n_clients']} | {r['series_per_client']} | {_n(r['total_s'], 1)} | {_n(r['prep_s'], 1)} | "
                     f"{_n(r['call_p50_s'])} | {_n(r['call_p95_s'])} | {_n(r['throughput_clients_per_s'], 1)} | "
                     f"{_n(r['peak_rss_mb'], 0)} | {_n(r.get('peak_rss_increase_mb'), 0)} |")
        L.append(f"\nRidge fit {_n(sv.get('ridge_fit_s'), 1)} s ({sv.get('ridge_unique_windows')} windows, {sv.get('ridge_scaling')} scaling); "
                 f"excluded from the modes: {sv.get('timing_excludes')}.")
    L += ["", "## E5d placement emulation (no model)", "",
          "| Policy | N0 | new/day | meters | P(co-located): new / all | share if co-located, new (median, p95) | "
          "isolated per run: 255 of 256 / whole block |", "|---|---|---|---|---|---|---|"]
    for p in s["placement"]:
        for r in p["rows"]:
            L.append(f"| {r['policy']} | {r['N0']} | {r['lam_per_day']:g} | {r['m']} | {_n(r['new_p_co_located'], 4)} / "
                     f"{_n(r['all_p_co_located'], 4)} | {_n(r['new_share_if_co_median'])} / {_n(r['new_share_if_co_p95'])} | "
                     f"{_n(r['all_isolated_255_per_run'], 2)} / {_n(r['all_isolated_any_per_run'], 2)} |")
        L += ["", "| Service policy N_est | k | attacker meters | P(reach a given victim) | occupancy |", "|---|---|---|---|---|"]
        for r in p["service_policy"]:
            L.append(f"| {r['N_est']} | {r['k']} | {r['m']} | {r['p_exact']:.4f} | {r['occupancy']:.4f} |")
    L += ["", f"Runtime of this report: {s['runtime_min']} min."]
    return "\n".join(L) + "\n"
